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
# Servidor de parámetros (SSP asíncrono)
# -------------------------------------------

class ParameterServer:
    def __init__(self, host: str, port: int, num_workers: int,
                 epochs: int, lr: float, batch_size: int,
                 quorum: int, ssp_bound: int):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.quorum = max(1, quorum)
        self.ssp_bound = max(0, ssp_bound)

        # Dispositivo / modelo / optimizador
        self.device = torch.device("cpu")
        self.model = Cifar10CNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Orden de parámetros (para alinear gradientes)
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_list = [p for _, p in self.model.named_parameters()]

        # Dataset (PS posee el dataset y envía lotes)
        self.transform_train = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        self.transform_test = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        self.train_set = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform_train
        )
        self.test_set = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform_test
        )

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

        # Buffer de contribuciones pendientes (no usadas aún)
        # Cada item: {rank, grads, batch_size, model_version, arrival}
        self.pending: List[Dict[str, Any]] = []
        self._arrival_ctr = 0

    # ---------- Épocas y batches ----------
    def _reset_epoch_iter(self):
        # dataloader "pull" para construir lotes on-demand y enviarlos en caliente
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=True,
            num_workers=2, pin_memory=False, drop_last=False
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
        test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=256, shuffle=False, num_workers=2)
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
        print(f"[PS] Epoch {epoch_idx}: Accuracy test = {acc:.2f}% | Tiempo = {et_str} | {ram_str}")

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
            # No hay más batches en esta época; el worker queda bloqueado esperando
            # hasta que inicie la siguiente época o finalice el entrenamiento.
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

        # Fin de época: si ya enviamos todos los batches y no hay trabajo en vuelo
        all_batches_dispatched = (self.batches_sent >= self.num_batches_per_epoch)
        # Considera solo workers vivos
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
                # Fin de entrenamiento
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
                            "local_steps": 0,           # #batches que procesó el worker
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
                            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        }
                        send_obj(conn, cfg)

                        # Primer envío de batch distinto a cada worker
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

                        # Comprobar staleness SSP: descartar si (global_step - versión_usada) > ssp_bound
                        stale = (self.global_step - model_version_used) > self.ssp_bound
                        if stale:
                            # Descarta contribución
                            print(f"[PS] DESCARTADO por SSP-bound: worker {worker_rank} "
                                  f"(usó step={model_version_used}, global={self.global_step}, ssp={self.ssp_bound})")
                        else:
                            # Bufferizar contribución para uso en updates presentes o futuros
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
                            # 1) purga primero lo que ya quedó fuera de SSP para siempre
                            self._purge_stale_pending()

                            # 2) luego intenta construir quórum con lo que sigue siendo válido
                            valid = [c for c in self.pending
                                     if (self.global_step - c["model_version"]) <= self.ssp_bound]
                            if len(valid) < self.quorum:
                                break

                            # Tomar las primeras 'quorum' por orden de llegada
                            valid.sort(key=lambda c: c["arrival"])
                            selected = valid[:self.quorum]

                            # Agregar gradientes ponderando por tamaño de batch
                            agg = [torch.zeros_like(p, device=self.device) for p in self.param_list]
                            total_bs = 0
                            for c in selected:
                                bs = c["batch_size"]
                                total_bs += bs
                                for i, g in enumerate(c["grads"]):
                                    agg[i] += g * bs
                            avg = [a / float(max(1, total_bs)) for a in agg]

                            # Aplicar update en el modelo
                            for p, g in zip(self.param_list, avg):
                                p.grad = g
                            self.optimizer.step()
                            self.optimizer.zero_grad(set_to_none=True)
                            self._samples_total += total_bs
                            self.global_step += 1
                            updated = True
                            print(f"[PS] UPDATE aplicado. Global step {self.global_step}")

                            # Quitar seleccionados del buffer (los no seleccionados quedan para steps futuros)
                            ids_sel = set(id(s) for s in selected)
                            self.pending = [c for c in self.pending if id(c) not in ids_sel]

                            # Purga extra tras avanzar el step global
                            self._purge_stale_pending()

                        # Tras procesar, enviar siguiente batch a ESTE worker (con piggyback de estado si hubo update)
                        if wk is not None:
                            # Si hubo update, obligamos a que este worker reciba el estado nuevo antes de entrenar
                            if updated:
                                wk["last_server_step"] = -1  # fuerza incluir state_dict en el próximo _send_train
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
    parser.add_argument("--num-workers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=1024,
                        help="Tamaño del lote que el PS enviará a los workers")
    parser.add_argument("--quorum", type=int, default=2,
                        help="Nº mínimo de contribuciones para actualizar el modelo")
    parser.add_argument("--ssp-bound", type=int, default=5,
                        help="Máxima obsolescencia permitida (en steps) para aceptar una contribución")
    args = parser.parse_args()

    
   

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        quorum=args.quorum,
        ssp_bound=args.ssp_bound
    )
    ps.start()


if __name__ == "__main__":
    main()
