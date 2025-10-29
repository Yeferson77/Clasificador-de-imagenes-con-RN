
import argparse
import socket
import struct
import threading
import pickle
import time
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
# Modelo CNN para CIFAR-10
# ---------------------------

class Cifar10CNN(nn.Module):
    """Red neuronal convolucional simple para clasificar CIFAR-10."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.5)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(100, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


# -------------------------------------------
# Servidor de parámetros con SSP (Stale Synchronous Parallel)
# -------------------------------------------

class ParameterServer:
    def __init__(self, host: str, port: int, num_workers: int,
                 epochs: int, steps_per_epoch: int, lr: float,
                 ssp_bound: int, quorum: int ):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.lr = lr

        # SSP
        self.ssp_bound = max(0, int(ssp_bound))
        self.quorum = max(1, int(quorum))
       

        # Modelo y optimizador
        self.device = torch.device("cpu")
        self.model = Cifar10CNN().to(self.device)
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

        # Reloj por worker para SSP (último step con el que contribuyó / reportó)
        self.worker_last_step: Dict[int, int] = {}  # rank -> last_step

        print(f"[PS] Modo SSP habilitado | ssp_bound = {self.ssp_bound} | quorum = {self.quorum}/{self.num_workers}")

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

    # ---------- SSP helpers ----------
    def _min_worker_step(self) -> int:
        if not self.worker_last_step:
            return 0
        return min(self.worker_last_step.values())

    def _would_violate_ssp_if_advance(self) -> bool:
        """¿Violaríamos el bound SSP si incrementamos global_step en +1?"""
        min_step = self._min_worker_step()
        next_gap = (self.global_step + 1) - min_step
        return next_gap > self.ssp_bound

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
                        self.worker_last_step[rank] = 0  # comenzará en step 0
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
                        "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "ssp_bound": self.ssp_bound,
                        "quorum": self.quorum,
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

                        # Actualizar reloj del worker
                        self.worker_last_step[worker_rank] = step

                        # Si el gradiente viene de un modelo más viejo que el global_step actual,
                        # bajo SSP descartamos y re-sincronizamos al worker.
                        if step < self.global_step:
                            resync = {
                                "type": "resync",
                                "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                                "step": self.global_step
                            }
                            send_obj(conn, resync)
                            continue

                        # Si por algún motivo el worker trae step > global (no debería), resync.
                        if step > self.global_step:
                            resync = {
                                "type": "resync",
                                "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                                "step": self.global_step
                            }
                            send_obj(conn, resync)
                            continue
                        
                        # Ignorar si el worker ya contribuyó en esta ronda
                        if worker_rank in self.contributors:
                            # Ya tenemos su contribución para self.global_step; mantenerlo esperando
                            # hasta que se aplique el update o se resync.
                            continue

                        # Acumular gradientes (para el step actual)
                        for i, g in enumerate(grads_list):
                            self.agg_grads_sum[i] += g.to(self.device) * batch_size
                        self.agg_samples += batch_size
                        self.waiting_socks.append(conn)
                        self.contributors.add(worker_rank)

                        # ¿Tenemos quórum suficiente para avanzar este step?
                        have_quorum = len(self.contributors) >= self.quorum

                        # ¿Respetamos el bound SSP si avanzamos ahora?
                        violates_ssp = self._would_violate_ssp_if_advance()

                        if have_quorum and not violates_ssp:
                            # Promedios y actualización
                            avg_grads = [g_sum / float(self.agg_samples) for g_sum in self.agg_grads_sum]

                            for p, g in zip(self.param_list, avg_grads):
                                p.grad = g
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)

                            self._samples_total += self.agg_samples
                            self.global_step += 1

                            print(f"[PS][SSP] UPDATE aplicado. "
                                  f"Step {self.global_step}/{self.total_steps} | "
                                  f"contribs={len(self.contributors)}/{self.num_workers} "
                                  f"(quorum={self.quorum})")

                            # Enviar actualización o stop a todos los workers que están esperando por este step
                            state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}
                            training_done = self.global_step >= self.total_steps

                            for s in self.waiting_socks:
                                if training_done:
                                    send_obj(s, {"type": "stop"})
                                else:
                                    send_obj(s, {"type": "update",
                                                 "state_dict": state_cpu,
                                                 "step": self.global_step})
                            
                            # Resetear estado de agregación para el nuevo step
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
                        else:
                            # No hay quórum o avanzar violaría SSP: mantener a los contribuyentes esperando.
                            if not have_quorum:
                                print(f"[PS][SSP] Esperando quórum: {len(self.contributors)}/{self.quorum} "
                                      f"en step {self.global_step}")
                            if violates_ssp:
                                min_step = self._min_worker_step()
                                print(f"[PS][SSP] Pausa por bound SSP: avanzar crearía gap "
                                      f"{(self.global_step + 1) - min_step} > {self.ssp_bound}. "
                                      f"(min_worker_step={min_step}, global_step={self.global_step})")

                elif mtype == "done":
                    print(f"[PS] Worker {addr} terminó y cerró la conexión.")
                    conn.close()
                    return

        except (ConnectionError, OSError):
            print(f"[PS] Conexión perdida con {addr}")
        finally:
            try:
                conn.close()
            except:
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

        print(f"[PS] Epoch {epoch_idx}: Accuracy test = {acc:.2f}% | "
              f"Tiempo = {et_str} | {ram_str}")
    
    # Evaluación final segura
    def _evaluate_on_test_with_state(self, state_dict_cpu: Dict[str, torch.Tensor]) -> float:
        self.model.load_state_dict({k: v.to(self.device) for k, v in state_dict_cpu.items()})
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        test_set = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=2)

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
    
    # Evaluación final y resumen robusto
    def evaluate_and_report(self):
        """Evaluación final del modelo + resumen robusto con join y lectura segura."""
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
            print(f"Tiempo total de entrenamiento: {total_time:.3f} s")
            print(f"Tiempo promedio por step: {avg_step_time * 1000:.2f} ms")
            print(f"Accuracy final (test): {acc_final:.2f}%\n")

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


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ssp-bound", type=int, default=3,
                        help="Máxima distancia permitida entre el step global y el mínimo step de los workers (SSP). 0 = totalmente síncrono.")
    parser.add_argument("--quorum", type=int, default=3,
                        help="Número mínimo de contribuciones para aplicar un update. Por defecto = num_workers - ssp_bound (min 1).")
    batch_size=128
    args = parser.parse_args()

    # steps_per_epoch original estaba dimensionado por num_workers. Mantenemos igual
    steps_per_epoch = math.floor((50000/args.num_workers)/batch_size)

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        lr=args.lr,
        ssp_bound=args.ssp_bound,
        quorum=args.quorum
    )
    ps.start()


if __name__ == "__main__":
    main()


# Ejecucion:

#python ps_server.py --num-workers 4 --ssp-bound 2 --quorum 3
