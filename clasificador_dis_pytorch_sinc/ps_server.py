import argparse
import socket
import struct
import threading
import pickle
import time
import os
import json
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T
import math

# ---- dependencia opcional para métricas de RAM ----
try:
    import psutil  # RAM (sistema y proceso)
except Exception:
    psutil = None


# ---------------------------
# Utilidades de red (sockets)
# ---------------------------

def send_obj(sock: socket.socket, obj: Any) -> None:
    """Envía un objeto Python serializado con pickle y longitud como prefijo."""
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket, n: int) -> bytes:
    """Lee exactamente n bytes de un socket."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Conexión cerrada por el cliente")
        buf.extend(chunk)
    return bytes(buf)


def recv_obj(sock: socket.socket) -> Any:
    """Recibe un objeto serializado con pickle y longitud como prefijo."""
    raw_len = recvall(sock, 8)
    (length,) = struct.unpack("!Q", raw_len)
    data = recvall(sock, length)
    return pickle.loads(data)


# ---------------------------
# Modelo CNN para ImageNet
# ---------------------------

class RN_Imagenet(nn.Module):
    """Red neuronal convolucional simple para clasificar ImageNet-1k."""
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),# 3x224x224 -> 32x224x224
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),# 32x112x112 -> 64x112x112
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),# 64x56x56 -> 128x56x56
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),# 128x28x28 -> 512x28x28
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# -------------------------------------------
# Servidor de parámetros (entrenamiento síncrono)
# -------------------------------------------

class ParameterServer:
    def __init__(self, host: str, port: int, num_workers: int,
                 epochs: int, steps_per_epoch: int, lr: float,
                 imagenet_root: str = "./data/imagenet"):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr
        self.imagenet_root = imagenet_root

        # Dispositivo
        self.device = torch.device("cpu")

        # Dataset de validación ImageNet (val) para evaluar
        self.transform_test = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        try:
            self.test_set = torchvision.datasets.ImageNet(
                root=self.imagenet_root,
                split="val",
                transform=self.transform_test
            )
        except (RuntimeError, FileNotFoundError) as e:
            raise RuntimeError(
                "No se encontró el dataset ImageNet en 'imagenet_root'. "
                "Estructura esperada: <root>/train y <root>/val con subcarpetas por clase. "
                f"Detalle: {e}"
            )

        self.num_classes = len(getattr(self.test_set, 'classes', [])) or 1000

        # Modelo y optimizador
        self.model = RN_Imagenet(num_classes=self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Lista de parámetros en orden
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_list = [p for _, p in self.model.named_parameters()]

        # Estado de entrenamiento
        self.global_step = 0
        self.total_steps = self.epochs * self.steps_per_epoch

        # Sincronización
        self.lock = threading.Lock()
        self.reset_aggregation_state()

        # Conexiones
        self.registered = 0
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        self.train_finished = False
        self.evaluated = False

        # Métricas de tiempo y accuracy
        self._t0_train: Optional[float] = None
        self._t_end_train: Optional[float] = None
        self._samples_total: int = 0

        self._epoch_times: List[float] = []
        self._epoch_start_t: Optional[float] = None
        self._epoch_accs: List[Optional[float]] = []

        # RAM por época (sistema + proceso)
        self._epoch_ram: List[Dict[str, Any]] = []

        # Hilos de evaluación por época (para join en el resumen final)
        self._epoch_eval_threads: List[threading.Thread] = []

        # Proceso para RAM del proceso (si psutil está disponible)
        self._proc = psutil.Process() if psutil else None

    def reset_aggregation_state(self):
        """Resetea los acumuladores para la ronda actual."""
        self.agg_grads_sum = [torch.zeros_like(p, device=self.device) for p in self.param_list]
        self.agg_samples = 0
        self.waiting_socks: List[socket.socket] = []
        self.contributors = set()

    def start(self):
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.num_workers)
        print(f"[PS] Escuchando en {self.host}:{self.port} (Esperando {self.num_workers} workers)")

        while True:
            conn, addr = self.server_sock.accept()
            print(f"[PS] Worker conectado desde {addr}")
            t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
            t.start()

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        try:
            while True:
                # Esperar mensaje del worker
                msg = recv_obj(conn)
                mtype = msg.get("type")

                # Manejo de registro
                if mtype == "register":
                    rank = msg["rank"]
                    world_size = msg["world_size"]

                    with self.lock:
                        self.registered += 1
                        print(f"[PS] Registro worker rank={rank}/{world_size} | "
                              f"Total registrados: {self.registered}/{self.num_workers}")

                    # Enviar configuración inicial
                    cfg = {
                        "type": "config",
                        "param_names": self.param_names,
                        "epochs": self.epochs,
                        "steps_per_epoch": self.steps_per_epoch,
                        "world_size": world_size,
                        "rank": rank,
                        "lr": self.lr,
                        "step": self.global_step,
                        "num_classes": self.num_classes,
                        "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                    }
                    send_obj(conn, cfg)

                # Manejo de gradientes recibidos
                elif mtype == "gradients":
                    worker_rank = msg["worker"]
                    step = msg["step"]
                    batch_size = int(msg["batch_size"])
                    grads_list = msg["grads"]

                    # Validaciones y agregación de gradientes
                    with self.lock:

                        # Marca tiempo de inicio de entrenamiento
                        if self._t0_train is None:
                            self._t0_train = time.perf_counter()
                            self._epoch_start_t = self._t0_train

                        # Ignorar si ya terminó el entrenamiento
                        if self.train_finished or self.global_step >= self.total_steps:
                            send_obj(conn, {"type": "stop"})
                            continue

                        if step != self.global_step:
                            # Worker desincronizado
                            print(f"[PS] Resync a worker {worker_rank}: "
                                  f"worker_step={step}, global_step={self.global_step}")
                            resync = {
                                "type": "resync",
                                "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                                "step": self.global_step
                            }
                            send_obj(conn, resync)
                            continue

                        # Ignorar si el worker ya contribuyó en esta ronda
                        if worker_rank in self.contributors:
                            continue

                        # Acumular gradientes
                        for i, g in enumerate(grads_list):
                            self.agg_grads_sum[i] += g.to(self.device) * batch_size
                        self.agg_samples += batch_size
                        self.waiting_socks.append(conn)
                        self.contributors.add(worker_rank)

                        # Si todos los workers contribuyeron, actualizar modelo
                        if len(self.contributors) == self.num_workers:
                            # Todos los workers enviaron gradientes
                            avg_grads = [g_sum / float(self.agg_samples) for g_sum in self.agg_grads_sum]

                            # Actualizar modelo
                            for p, g in zip(self.param_list, avg_grads):
                                p.grad = g
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                            self._samples_total += self.agg_samples
                            self.global_step += 1

                            print(f"[PS] UPDATE aplicado. Global step {self.global_step}/{self.total_steps}")

                            # Enviar actualización o stop a todos los workers
                            state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
                            training_done = self.global_step >= self.total_steps

                            for s in self.waiting_socks:
                                if training_done:
                                    send_obj(s, {"type": "stop"})
                                else:
                                    send_obj(s, {"type": "update",
                                                 "state_dict": state_cpu,
                                                 "step": self.global_step})

                            # Resetear estado de agregación
                            self.reset_aggregation_state()

                            # Fin de época
                            if self.global_step % self.steps_per_epoch == 0:
                                epoch_idx = self.global_step // self.steps_per_epoch
                                now = time.perf_counter()
                                if self._epoch_start_t is not None:
                                    self._epoch_times.append(now - self._epoch_start_t)
                                self._epoch_start_t = now

                                # Captura de RAM por época
                                self._epoch_ram.append(self._sample_ram_metrics())

                                # Lanzar evaluación por-época y guardar el hilo (para join)
                                t = threading.Thread(
                                    target=self._eval_epoch_snapshot,
                                    args=(state_cpu, epoch_idx),
                                    daemon=True
                                )
                                t.start()
                                self._epoch_eval_threads.append(t)
                            # Fin de entrenamiento
                            if training_done and not self.train_finished:
                                self.train_finished = True
                                self._t_end_train = time.perf_counter()
                                threading.Thread(target=self.evaluate_and_report, daemon=True).start()

                elif mtype == "done":
                    print(f"[PS] Worker {addr} terminó y cerró la conexión.")
                    conn.close()
                    return

        except (ConnectionError, OSError):
            print(f"[PS] Conexión perdida con {addr}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # --------- RAM (sistema y proceso) ----------
    def _sample_ram_metrics(self) -> Dict[str, Any]:
        """
        Devuelve un dict con:
        - ram_total_gb
        - ram_used_gb
        - ram_percent
        - proc_rss_gb  (RAM usada por este proceso)
        """
        out: Dict[str, Any] = {"ram_total_gb": "N/A", "ram_used_gb": "N/A",
                               "ram_percent": "N/A", "proc_rss_gb": "N/A"}
        if psutil:
            try:
                vm = psutil.virtual_memory()
                out["ram_total_gb"] = round(vm.total / (1024 ** 3), 3)
                out["ram_used_gb"] = round((vm.total - vm.available) / (1024 ** 3), 3)
                out["ram_percent"] = vm.percent
                if self._proc:
                    rss = self._proc.memory_info().rss
                    out["proc_rss_gb"] = round(rss / (1024 ** 3), 3)
            except Exception:
                pass
        return out

    # Evaluación por-época en hilo separado
    def _eval_epoch_snapshot(self, state_dict_cpu: Dict[str, torch.Tensor], epoch_idx: int):
        acc = self._evaluate_on_test_with_state(state_dict_cpu)
        with self.lock:
            self._epoch_accs.append(acc)
            # Asegura que la lista de accuracies no quede atrás si faltara alguna época
            while len(self._epoch_accs) < len(self._epoch_times):
                self._epoch_accs.append(None)

        # Log robusto por época
        try:
            et = self._epoch_times[-1]
        except Exception:
            et = None
        et_str = f"{et:.3f}s" if et is not None else "N/A"

        # RAM de la última época
        ram = self._epoch_ram[-1] if self._epoch_ram else {}
        ram_str = (f"RAM(sys): {ram.get('ram_used_gb','N/A')}/{ram.get('ram_total_gb','N/A')} GB "
                   f"({ram.get('ram_percent','N/A')}%) | "
                   f"RAM(proc): {ram.get('proc_rss_gb','N/A')} GB")

        print(f"[PS] Epoch {epoch_idx}: Accuracy val (ImageNet) = {acc:.2f}% | "
              f"Tiempo = {et_str} | {ram_str}")

    # Evaluación final segura
    def _evaluate_on_test_with_state(self, state_dict_cpu: Dict[str, torch.Tensor]) -> float:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict_cpu.items()})
        test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=256, shuffle=False, num_workers=4
        )

        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()

        return 100.0 * correct / max(total, 1)

    # Evaluación final y resumen robusto (incluye JSON en ./informes)
    def evaluate_and_report(self):
        """Evaluación final del modelo + resumen robusto con join y escritura a JSON."""
        if self.evaluated:
            return
        self.evaluated = True

        # esperar a que terminen los hilos por-época
        for t in getattr(self, "_epoch_eval_threads", []):
            if t.is_alive():
                t.join()

        # Accuracy final con el estado actual del modelo
        acc_final = self._evaluate_on_test_with_state({k: v.cpu() for k, v in self.model.state_dict().items()})

        if self._t0_train and self._t_end_train:
            total_time = self._t_end_train - self._t0_train
            avg_step_time = total_time / max(self.total_steps, 1)

            print("\n========== RESUMEN FINAL ==========")
            print(f"Workers utilizados: {self.num_workers}")
            print(f"Épocas totales: {self.epochs}")
            print(f"Steps por época: {self.steps_per_epoch}")
            print(f"Steps totales (planificados): {self.total_steps}")
            print(f"Tiempo total de entrenamiento: {total_time:.3f} s")
            print(f"Tiempo promedio por step: {avg_step_time * 1000:.2f} ms")
            print(f"Accuracy final val (ImageNet): {acc_final:.2f}%\n")

            print("---- Métricas por época ----")
            epoch_metrics: List[Dict[str, Any]] = []
            with self.lock:
                total_epochs = max(len(self._epoch_times), len(self._epoch_accs), len(self._epoch_ram))
                for i in range(1, total_epochs + 1):
                    idx = i - 1
                    tsec = self._epoch_times[idx] if idx < len(self._epoch_times) else None
                    acc_ep = self._epoch_accs[idx] if idx < len(self._epoch_accs) else None
                    ram = self._epoch_ram[idx] if idx < len(self._epoch_ram) else {}

                    tsec_str = f"{tsec:.3f}s" if tsec is not None else "N/A"
                    acc_str = f"{acc_ep:.2f}%" if (acc_ep is not None) else "N/A"
                    ram_str = (f"RAM(sys): {ram.get('ram_used_gb','N/A')}/{ram.get('ram_total_gb','N/A')} GB "
                               f"({ram.get('ram_percent','N/A')}%) | "
                               f"RAM(proc): {ram.get('proc_rss_gb','N/A')} GB") if ram else "RAM: N/A"

                    print(f"Época {i:02d}: Tiempo = {tsec_str} | Accuracy = {acc_str} | {ram_str}")

                    epoch_metrics.append({
                        "epoch": i,
                        "time_seconds": tsec,
                        "accuracy": acc_ep,
                        "ram": ram,
                    })

            # ---- Tasa de convergencia segura ----
            conv_rates = []
            with self.lock:
                for i in range(1, len(self._epoch_accs)):
                    prev, cur = self._epoch_accs[i - 1], self._epoch_accs[i]
                    if prev is None or cur is None:
                        continue
                    delta = cur - prev
                    rel = (delta / prev * 100.0) if prev != 0 else float('inf')
                    conv_rates.append((i + 1, delta, rel))

            if conv_rates:
                print("\n---- Tasa de convergencia ----")
                for ep, delta, rel in conv_rates:
                    print(f"Época {ep:02d}: Δacc = {delta:+.2f} pts | Δ% = {rel:+.2f}%")
            print("====================================\n")

            # ---- Guardar resumen en JSON en carpeta 'informes' ----
            resumen_json = {
                "workers": self.num_workers,
                "epochs": self.epochs,
                "steps_per_epoch": self.steps_per_epoch,
                "total_steps_planned": self.total_steps,
                "global_steps_executed": self.global_step,
                "total_training_time_seconds": total_time,
                "avg_step_time_seconds": avg_step_time,
                "final_val_accuracy": acc_final,
                "epoch_metrics": epoch_metrics,
            }

            try:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                informes_dir = os.path.join(script_dir, "informes")
                os.makedirs(informes_dir, exist_ok=True)
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                json_path = os.path.join(informes_dir, f"reporte_final_imagenet_sync_{timestamp}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(resumen_json, f, ensure_ascii=False, indent=4)
                print(f"[PS] Resumen final guardado en JSON: {json_path}")
            except Exception as e:
                print(f"[PS] Error guardando resumen JSON en 'informes': {e}")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--imagenet-root", type=str, default="../clasificador_dis_pytorch_asinc_imagenet/data/imagenet",
                        help="Ruta a la raíz que contiene los splits 'train' y 'val' de ILSVRC2012")
    batch_size = 32
    args = parser.parse_args()

    # Tamaño aproximado del split de entrenamiento de ImageNet-1k
    imagenet_train_size = 1281167 * 0.15  # Usando 3% del dataset completo
    steps_per_epoch = math.floor((imagenet_train_size / args.num_workers) / batch_size)

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        lr=args.lr,
        imagenet_root=args.imagenet_root,
    )
    ps.start()


if __name__ == "__main__":
    main()

# Ejemplo:
# python ps_server.py --host 127.0.0.1 --port 5000 --num-workers 2 --epochs 5 --lr 0.001 --imagenet-root ./data/imagenet
