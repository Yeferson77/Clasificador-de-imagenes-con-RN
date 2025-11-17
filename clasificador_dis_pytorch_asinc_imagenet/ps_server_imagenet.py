import argparse
import socket
import struct
import threading
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T

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
# Modelo CNN (input-size agnóstico con AdaptiveAvgPool2d)
# ---------------------------

class Cifar10CNN(nn.Module):
    """CNN simple ajustada para ImageNet (1k clases) sin forzar 32x32.
       Usa AdaptiveAvgPool2d para soportar entradas 224x224 (pipeline estándar de ImageNet).
    """
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 224 -> 112

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112 -> 56

            nn.Dropout(0.5),
            # Bloque extra para ganar capacidad con 224x224
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28
            nn.Dropout(0.5),

            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (C,1,1)
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
# Servidor de parámetros (SSP asíncrono)
# -------------------------------------------

class ParameterServer:
    def __init__(self, host: str, port: int, num_workers: int,
                 epochs: int, lr: float, batch_size: int,
                 quorum: int, ssp_bound: int,
                 imagenet_root: str = "./data/imagenet"):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.quorum = max(1, quorum)
        self.ssp_bound = max(0, ssp_bound)

        # Dispositivo
        self.device = torch.device("cpu")

        # ================================
        # Dataset: ILSVRC / ImageNet-1k
        # Usamos torchvision.datasets.ImageNet
        # Pipeline estándar (224):
        #   - Train: RandomResizedCrop(224), RandomHorizontalFlip
        #   - Val: Resize(256) + CenterCrop(224)
        # ================================
        self.transform_train = T.Compose([
            T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        self.transform_test = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
        ])
        try:
            self.train_set = torchvision.datasets.ImageNet(
                root=imagenet_root, split="train", transform=self.transform_train
            )
            self.test_set = torchvision.datasets.ImageNet(
                root=imagenet_root, split="val", transform=self.transform_test
            )
        except (RuntimeError, FileNotFoundError) as e:
            raise RuntimeError(
                "No se encontró el dataset ImageNet en 'imagenet_root'. "
                "Estructura esperada: <root>/train y <root>/val con subcarpetas por clase. "
                f"Detalle: {e}"
            )

        self.num_classes = len(getattr(self.train_set, 'classes', [])) or 1000

        # Modelo / optimizador
        self.model = Cifar10CNN(num_classes=self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Orden de parámetros (para alinear gradientes)
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_list = [p for _, p in self.model.named_parameters()]

        self.num_batches_per_epoch = math.ceil(len(self.train_set) / self.batch_size)

        # Estado de entrenamiento
        self.global_step = 0             # pasos de actualización del modelo (no = #batches)
        self.epoch_idx = 0               # 1..epochs
        self.train_finished = False
        self.evaluated = False

        self._samples_total = 0
        self._t0_train: Optional[float] = None
        self._t_end_train: Optional[float] = None

        # Métricas por época
        self._epoch_start_t: Optional[float] = None
        self._epoch_times: List[float] = []
        self._epoch_accs: List[Optional[float]] = []
        self._epoch_ram: List[Dict[str, Any]] = []
        self._epoch_eval_threads: List[threading.Thread] = []

        # RAM proceso (si psutil)
        self._proc = psutil.Process() if psutil else None

        # Sincronización
        self.lock = threading.Lock()

        # Conexiones / workers
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Estado de workers: rank -> dict
        self.workers: Dict[int, Dict[str, Any]] = {}
        self.registered = 0

        # Batching por época
        self._reset_epoch_iter()

        # Buffer de contribuciones pendientes
        self.pending: List[Dict[str, Any]] = []
        self._arrival_ctr = 0

    # ---------- Épocas y batches ----------
    def _reset_epoch_iter(self):
        # dataloader "pull" para construir lotes on-demand y enviarlos en caliente
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=4, pin_memory=False, drop_last=False
        )
        self.train_iter = iter(self.train_loader)
        self.batches_sent = 0  # enviados en esta época

    def _next_batch(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        try:
            x, y = next(self.train_iter)
            self.batches_sent += 1
            return x, y
        except StopIteration:
            return None

    # ---------- RAM (sistema y proceso) ----------
    def _sample_ram_metrics(self) -> Dict[str, Any]:
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

    # ---------- Evaluación ----------
    def _evaluate_on_test_with_state(self, state_dict_cpu: Dict[str, torch.Tensor]) -> float:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict_cpu.items()})
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=128, shuffle=False, num_workers=4)
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
        return 100.0 * correct / total

    def _eval_epoch_snapshot(self, state_dict_cpu: Dict[str, torch.Tensor], epoch_idx: int):
        acc = self._evaluate_on_test_with_state(state_dict_cpu)
        with self.lock:
            self._epoch_accs.append(acc)
            while len(self._epoch_accs) < len(self._epoch_times):
                self._epoch_accs.append(None)
        try:
            et = self._epoch_times[-1]
        except Exception:
            et = None
        et_str = f"{et:.3f}s" if et is not None else "N/A"
        ram = self._epoch_ram[-1] if self._epoch_ram else {}
        ram_str = (f"RAM(sys): {ram.get('ram_used_gb','N/A')}/{ram.get('ram_total_gb','N/A')} GB "
                   f"({ram.get('ram_percent','N/A')}%) | "
                   f"RAM(proc): {ram.get('proc_rss_gb','N/A')} GB")
        print(f"[PS] Epoch {epoch_idx}: Accuracy val (ImageNet) = {acc:.2f}% | Tiempo = {et_str} | {ram_str}")

    def evaluate_and_report(self):
        if self.evaluated:
            return
        self.evaluated = True
        for t in getattr(self, "_epoch_eval_threads", []):
            if t.is_alive():
                t.join()
        acc_final = self._evaluate_on_test_with_state({k: v.cpu() for k, v in self.model.state_dict().items()})

        if self._t0_train and self._t_end_train:
            total_time = self._t_end_train - self._t0_train
            avg_step_time = total_time / max(self.global_step, 1)
            print("\n========== RESUMEN FINAL ==========")
            print(f"Workers utilizados: {self.num_workers}")
            print(f"Épocas totales: {self.epochs}")
            print(f"Pasos (updates) globales: {self.global_step}")
            print(f"Tiempo total de entrenamiento: {total_time:.3f} s")
            print(f"Tiempo promedio por update: {avg_step_time * 1000:.2f} ms")
            print(f"Accuracy final val (ImageNet): {acc_final:.2f}%\n")

            print("---- Métricas por época ----")
            with self.lock:
                total_epochs = max(len(self._epoch_times), len(self._epoch_accs), len(self._epoch_ram))
                for i in range(1, total_epochs + 1):
                    tsec = self._epoch_times[i - 1] if i - 1 < len(self._epoch_times) else None
                    acc_ep = self._epoch_accs[i - 1] if i - 1 < len(self._epoch_accs) else None
                    ram = self._epoch_ram[i - 1] if i - 1 < len(self._epoch_ram) else {}
                    tsec_str = f"{tsec:.3f}s" if tsec is not None else "N/A"
                    acc_str = f"{acc_ep:.2f}%" if (acc_ep is not None) else "N/A"
                    ram_str = (f"RAM(sys): {ram.get('ram_used_gb','N/A')}/{ram.get('ram_total_gb','N/A')} GB "
                               f"({ram.get('ram_percent','N/A')}%) | "
                               f"RAM(proc): {ram.get('proc_rss_gb','N/A')} GB") if ram else "RAM: N/A"
                    print(f"Época {i:02d}: Tiempo = {tsec_str} | Accuracy = {acc_str} | {ram_str}")
            print("====================================\n")

    # ---------- Utilidad SSP: purgar aportes irrecuperables ----------
    def _purge_stale_pending(self):
        """Elimina del buffer contribuciones que ya nunca podrán usarse por SSP-bound."""
        before = len(self.pending)
        self.pending = [
            c for c in self.pending
            if (self.global_step - c["model_version"]) <= self.ssp_bound
        ]
        removed = before - len(self.pending)
        if removed > 0:
            print(f"[PS] Purged {removed} stale contributions (ssp={self.ssp_bound}, global={self.global_step})")

    # ---------- Ciclo de servidor ----------
    def start(self):
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.num_workers)
        print(f"[PS] Escuchando en {self.host}:{self.port} (Esperando {self.num_workers} workers)")

        # marca inicio de entrenamiento
        self._t0_train = time.perf_counter()
        self._epoch_start_t = self._t0_train
        self.epoch_idx = 1
        print(f"[PS] Iniciando Época {self.epoch_idx}/{self.epochs} | "
              f"batches/época={self.num_batches_per_epoch} | quorum={self.quorum} | ssp={self.ssp_bound}")

        while True:
            conn, addr = self.server_sock.accept()
            t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
            t.start()

    # ---------- Envío de batch + (opcional) nuevo estado ----------
    def _send_train(self, rank: int, conn: socket.socket):
        """Envía un batch a un worker. Si no hay batch disponible (época agotada), no envía nada."""
        # Si ya terminó todo el entrenamiento, enviamos stop
        if self.train_finished:
            try:
                send_obj(conn, {"type": "stop"})
            except Exception:
                pass
            return

        # Obtener siguiente batch; si no hay, esperar a que cambie de época o finalice
        batch = self._next_batch()
        if batch is None:
            return

        x, y = batch
        # Si el worker está desactualizado, piggyback del estado
        state_cpu = None
        if self.workers[rank]["last_server_step"] < self.global_step:
            state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
            self.workers[rank]["last_server_step"] = self.global_step

        msg = {
            "type": "train",
            "epoch": self.epoch_idx,
            "step": self.global_step,
            "batch_id": self.batches_sent,
            "data": x.cpu(),
            "targets": y.cpu(),
            "state_dict": state_cpu  # puede ser None si el worker ya está al día
        }
        send_obj(conn, msg)
        self.workers[rank]["busy"] = True

    def _maybe_finish_epoch_or_training(self):
        """Detecta fin de época y programa la siguiente, o fin de entrenamiento."""
        # Purga contribuciones que ya nunca podrán usarse
        self._purge_stale_pending()

        all_batches_dispatched = (self.batches_sent >= self.num_batches_per_epoch)
        all_idle = all((not w.get("busy", False)) for w in self.workers.values() if w.get("alive", False))

        if all_batches_dispatched and all_idle:
            # Cierre de la época actual
            now = time.perf_counter()
            if self._epoch_start_t is not None:
                self._epoch_times.append(now - self._epoch_start_t)
            self._epoch_start_t = now
            self._epoch_ram.append(self._sample_ram_metrics())

            # Evaluación snapshot
            state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
            t = threading.Thread(target=self._eval_epoch_snapshot, args=(state_cpu, self.epoch_idx), daemon=True)
            t.start()
            self._epoch_eval_threads.append(t)

            if self.epoch_idx >= self.epochs:
                self.train_finished = True
                self._t_end_train = time.perf_counter()
                # Notificar parada a todos
                for rk, w in list(self.workers.items()):
                    try:
                        send_obj(w["conn"], {"type": "stop"})
                    except Exception:
                        pass
                threading.Thread(target=self.evaluate_and_report, daemon=True).start()
                return

            # Empezar nueva época
            self.epoch_idx += 1
            print(f"[PS] Iniciando Época {self.epoch_idx}/{self.epochs}")
            self._reset_epoch_iter()

            # Envío de primer batch distinto a cada worker disponible
            for rk, w in self.workers.items():
                if w.get("alive", False):
                    try:
                        self._send_train(rk, w["conn"])
                    except Exception:
                        pass

    # ---------- Manejo de cada cliente ----------
    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        rank: Optional[int] = None
        try:
            while True:
                msg = recv_obj(conn)
                mtype = msg.get("type")

                if mtype == "register":
                    rank = int(msg["rank"])
                    world_size = int(msg["world_size"])
                    with self.lock:
                        self.registered += 1
                        self.workers[rank] = {
                            "conn": conn,
                            "busy": False,
                            "local_steps": 0,
                            "last_server_step": self.global_step,
                            "alive": True,
                        }
                        print(f"[PS] Registro worker rank={rank}/{world_size} | "
                              f"Total registrados: {self.registered}/{self.num_workers}")

                        # Enviar config inicial + estado del modelo
                        cfg = {
                            "type": "config",
                            "param_names": self.param_names,
                            "epochs": self.epochs,
                            "lr": self.lr,
                            "batch_size": self.batch_size,
                            "quorum": self.quorum,
                            "ssp_bound": self.ssp_bound,
                            "step": self.global_step,
                            "num_classes": self.num_classes,
                            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        }
                        send_obj(conn, cfg)

                        # Primer envío de batch
                        self._send_train(rank, conn)

                elif mtype == "gradients":
                    worker_rank = int(msg["worker"])
                    model_version_used = int(msg["model_step_used"])  # versión del modelo que el worker usó
                    batch_sz = int(msg["batch_size"])
                    grads_list: List[torch.Tensor] = msg["grads"]

                    with self.lock:
                        if self.train_finished:
                            send_obj(conn, {"type": "stop"})
                            continue

                        # Marca inicio (si no estaba)
                        if self._t0_train is None:
                            self._t0_train = time.perf_counter()
                            self._epoch_start_t = self._t0_train

                        # Progreso local del worker
                        wk = self.workers.get(worker_rank)
                        if wk is not None:
                            wk["local_steps"] += 1
                            wk["busy"] = False

                        # Comprobar staleness SSP
                        stale = (self.global_step - model_version_used) > self.ssp_bound
                        if stale:
                            print(f"[PS] DESCARTADO por SSP-bound: worker {worker_rank} "
                                  f"(usó step={model_version_used}, global={self.global_step}, ssp={self.ssp_bound})")
                        else:
                            # Bufferizar contribución
                            self._arrival_ctr += 1
                            self.pending.append({
                                "rank": worker_rank,
                                "grads": [g.to(self.device) for g in grads_list],
                                "batch_size": batch_sz,
                                "model_version": model_version_used,
                                "arrival": self._arrival_ctr
                            })

                        # Intentar aplicar updates mientras haya quórum
                        updated = False
                        while True:
                            self._purge_stale_pending()

                            valid = [c for c in self.pending
                                     if (self.global_step - c["model_version"]) <= self.ssp_bound]
                            if len(valid) < self.quorum:
                                break

                            valid.sort(key=lambda c: c["arrival"])
                            selected = valid[:self.quorum]

                            agg = [torch.zeros_like(p, device=self.device) for p in self.param_list]
                            total_bs = 0
                            for c in selected:
                                bs = c["batch_size"]
                                total_bs += bs
                                for i, g in enumerate(c["grads"]):
                                    agg[i] += g * bs
                            avg = [a / float(max(1, total_bs)) for a in agg]

                            for p, g in zip(self.param_list, avg):
                                p.grad = g
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self._samples_total += total_bs
                            self.global_step += 1
                            updated = True
                            print(f"[PS] UPDATE aplicado. Global step {self.global_step}")

                            ids_sel = set(id(s) for s in selected)
                            self.pending = [c for c in self.pending if id(c) not in ids_sel]

                            self._purge_stale_pending()

                        # Tras procesar, enviar siguiente batch a ESTE worker
                        if wk is not None:
                            if updated:
                                wk["last_server_step"] = -1  # forzar piggyback de estado nuevo
                            self._send_train(worker_rank, wk["conn"])

                        # Ver si se terminó la época o el entrenamiento
                        self._maybe_finish_epoch_or_training()

                elif mtype == "done":
                    with self.lock:
                        if rank is not None and rank in self.workers:
                            self.workers[rank]["alive"] = False
                    conn.close()
                    return

        except (ConnectionError, OSError):
            with self.lock:
                if rank is not None and rank in self.workers:
                    self.workers[rank]["alive"] = False
            print(f"[PS] Conexión perdida con {addr}")
        finally:
            try:
                conn.close()
            except Exception:
                pass


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Tamaño del lote que el PS enviará a los workers (224x224 es pesado por red)")
    parser.add_argument("--quorum", type=int, default=2,
                        help="Nº mínimo de contribuciones para actualizar el modelo")
    parser.add_argument("--ssp-bound", type=int, default=2,
                        help="Máxima obsolescencia permitida (en steps) para aceptar una contribución")
    parser.add_argument("--imagenet-root", type=str, default="./data/imagenet",
                        help="Ruta a la raíz que contiene los splits 'train' y 'val' de ILSVRC2012")
    args = parser.parse_args()

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        quorum=args.quorum,
        ssp_bound=args.ssp_bound,
        imagenet_root=args.imagenet_root
    )
    ps.start()


if __name__ == "__main__":
    main()
