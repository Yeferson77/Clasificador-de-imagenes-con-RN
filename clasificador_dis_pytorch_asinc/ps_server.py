import argparse
import socket
import struct
import threading
import pickle
import time
from typing import List, Dict, Any, Tuple, Optional, Deque
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader

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
# Servidor de parámetros con SSP + envío de lotes (PS posee el dataset)
# -------------------------------------------

class ParameterServer:
    def __init__(self, host: str, port: int, num_workers: int,
                 epochs: int, batch_size: int, lr: float,
                 ssp_bound: int, quorum: Optional[int] = None,
                 num_workers_loader: int = 2, pin_memory: bool = False):
        self.host = host
        self.port = port
        self.num_workers = num_workers
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        # SSP
        self.ssp_bound = max(0, int(ssp_bound))
        default_quorum = self.num_workers - self.ssp_bound
        if quorum is None:
            self.quorum = max(1, min(self.num_workers, default_quorum))
        else:
            self.quorum = max(1, min(self.num_workers, int(quorum)))

        # Modelo y optimizador
        self.device = torch.device("cpu")
        self.model = Cifar10CNN().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # Lista de parámetros en orden
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.param_list = [p for _, p in self.model.named_parameters()]

        # Estado de entrenamiento
        self.global_step = 0

        # Sincronización
        self.lock = threading.Lock()

        # Conexiones
        self.server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Dataset / Dataloader (en el PS)
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                        std=(0.2470, 0.2435, 0.2616)),
        ])
        self.train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        # No usamos shuffle aquí; barajamos por-época manualmente con DataLoader y re-creamos cada época
        self.num_workers_loader = num_workers_loader
        self.pin_memory = pin_memory

        self.epoch_idx = 0
        self._build_epoch_loader()

        # Gestión de workers
        self.clients: Dict[int, socket.socket] = {}     # rank -> socket
        self.worker_busy: Dict[int, bool] = {}          # rank -> está ocupado con un batch
        self.worker_last_step: Dict[int, int] = {}      # rank -> último step completado

        # Pendientes
        self.pending_grads: Deque[Dict[str, Any]] = deque()   # cola de gradientes pendientes (orden llegada)

        # Métricas
        self._t0_train: Optional[float] = None
        self._t_end_train: Optional[float] = None
        self.total_batches_sent = 0
        self.total_batches_used = 0
        self.total_batches_discarded = 0
        self.total_updates = 0

        print(f"[PS] SSP dinámico habilitado | ssp_bound={self.ssp_bound} | quorum={self.quorum}/{self.num_workers} | batch_size={self.batch_size}")

    # ---------------------------
    # Loader por-época
    # ---------------------------
    def _build_epoch_loader(self):
        # Re-crea el DataLoader con shuffle para cada época
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                       shuffle=True, drop_last=False,
                                       num_workers=self.num_workers_loader,
                                       pin_memory=self.pin_memory)
        self.train_iter = iter(self.train_loader)
        self.batches_in_epoch = 0  # Contador para info
        print(f"[PS] Epoch {self.epoch_idx+1}/{self.epochs} inicializada.")

    # ---------------------------
    # Networking
    # ---------------------------
    def start(self):
        self.server_sock.bind((self.host, self.port))
        self.server_sock.listen(self.num_workers)
        print(f"[PS] Escuchando en {self.host}:{self.port} (esperando {self.num_workers} workers)")

        # Aceptar conexiones entrantes
        while True:
            conn, addr = self.server_sock.accept()
            t = threading.Thread(target=self.handle_client, args=(conn, addr), daemon=True)
            t.start()

    def handle_client(self, conn: socket.socket, addr: Tuple[str, int]):
        try:
            while True:
                msg = recv_obj(conn)
                mtype = msg.get("type")

                if mtype == "register":
                    rank = msg["rank"]
                    world_size = msg["world_size"]
                    with self.lock:
                        self.clients[rank] = conn
                        self.worker_busy[rank] = False
                        self.worker_last_step[rank] = -1  # aún no ha contribuido
                        print(f"[PS] Registro worker rank={rank}/{world_size}. Total={len(self.clients)}/{self.num_workers}")
                    # Enviar config inicial (sin dataset, el PS enviará lotes)
                    cfg = {
                        "type": "config",
                        "param_names": self.param_names,
                        "epochs": self.epochs,
                        "world_size": world_size,
                        "rank": rank,
                        "lr": self.lr,
                        "step": self.global_step,
                        "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
                        "ssp_bound": self.ssp_bound,
                        "quorum": self.quorum,
                        "batch_from_server": True,
                    }
                    send_obj(conn, cfg)
                    # Si ya tenemos suficientes workers, asignar lotes iniciales
                    with self.lock:
                        self._maybe_kickoff_initial_work()

                elif mtype == "ready":
                    rank = msg["rank"]
                    with self.lock:
                        self.worker_busy[rank] = False
                        self._assign_next_batch_if_available(rank)

                elif mtype == "gradients":
                    # Llega un gradiente; lo encolamos (o descartamos si excede staleness) y asignamos más trabajo
                    worker_rank = msg["worker"]
                    step_used = msg["step"]
                    batch_size = int(msg["batch_size"])
                    grads_list = msg["grads"]

                    with self.lock:
                        if self._t0_train is None:
                            self._t0_train = time.perf_counter()

                        # Marcar worker libre
                        self.worker_busy[worker_rank] = False
                        self.worker_last_step[worker_rank] = max(self.worker_last_step[worker_rank], step_used)

                        # Descartar si muy viejo respecto al step actual
                        if step_used < self.global_step - self.ssp_bound:
                            self.total_batches_discarded += 1
                            print(f"[PS][SSP] DESCARTA gradiente de step {step_used} (global={self.global_step}, bound={self.ssp_bound}).")
                        else:
                            self.pending_grads.append({
                                "worker": worker_rank,
                                "step": step_used,
                                "batch_size": batch_size,
                                "grads": [g.to(self.device) for g in grads_list]
                            })

                        # Intentar aplicar tantas actualizaciones como sea posible
                        self._apply_updates_if_possible()

                        # Asignar más trabajo al worker que acaba de enviar
                        self._assign_next_batch_if_available(worker_rank)

                        # ¿terminamos dataset y pendientes?
                        if self._is_training_finished():
                            self._finalize_and_stop()
                            return

                elif mtype == "done":
                    print(f"[PS] Worker {addr} cerró la conexión.")
                    conn.close()
                    return
        except (ConnectionError, OSError):
            print(f"[PS] Conexión perdida con {addr}")
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ---------------------------
    # Lógica de asignación / updates
    # ---------------------------
    def _assign_next_batch_if_available(self, rank: int):
        """Si hay batch disponible, enviarlo al worker 'rank' con el step actual y pesos actuales."""
        if rank not in self.clients:
            return
        if self.worker_busy.get(rank, True):
            return

        try:
            x, y = next(self.train_iter)
        except StopIteration:
            # Fin de la época actual
            self.epoch_idx += 1
            if self.epoch_idx >= self.epochs:
                # No quedan más batches para enviar
                return
            self._build_epoch_loader()
            try:
                x, y = next(self.train_iter)
            except StopIteration:
                return  # dataset vacío (no debería)

        # Preparar paquete de trabajo
        batch = {
            "type": "work",
            "step": self.global_step,
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},  # enviamos pesos actuales
            "x": x.cpu(),
            "y": y.cpu(),
        }
        send_obj(self.clients[rank], batch)
        self.worker_busy[rank] = True
        self.total_batches_sent += 1
        self.batches_in_epoch += 1

    def _maybe_kickoff_initial_work(self):
        """Al registrar workers, si están libres, envía un primer batch a cada uno."""
        for r in list(self.clients.keys()):
            self._assign_next_batch_if_available(r)

    def _apply_updates_if_possible(self):
        """
        Mientras haya al menos 'quorum' gradientes elegibles (step >= global_step - ssp_bound),
        aplica un update usando las primeras 'quorum' contribuciones en orden de llegada.
        """
        made_update = True
        while made_update:
            made_update = False

            # recolectar candidatos respetando bound, en orden de llegada
            eligibles_idx = []
            for idx, gmsg in enumerate(self.pending_grads):
                if gmsg["step"] >= self.global_step - self.ssp_bound:
                    eligibles_idx.append(idx)
                    if len(eligibles_idx) == self.quorum:
                        break

            if len(eligibles_idx) >= self.quorum:
                # extraer en el mismo orden y aplicar
                grads_take = [self.pending_grads[i] for i in eligibles_idx]
                # eliminar de la deque por índices (de atrás hacia adelante)
                for i in reversed(eligibles_idx):
                    self.pending_grads.remove(self.pending_grads[i])

                # Promedio ponderado por tamaño de batch
                total_bs = sum(g["batch_size"] for g in grads_take)
                avg_grads = [torch.zeros_like(p, device=self.device) for p in self.param_list]
                for gmsg in grads_take:
                    w = gmsg["batch_size"] / float(total_bs)
                    for i, g in enumerate(gmsg["grads"]):
                        avg_grads[i] += w * g

                # Aplicar
                for p, g in zip(self.param_list, avg_grads):
                    p.grad = g
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

                self.total_updates += 1
                self.total_batches_used += total_bs
                self.global_step += 1

                print(f"[PS][SSP] UPDATE aplicado -> step={self.global_step} | "
                      f"consumidas={self.quorum} contribs | "
                      f"pendientes={len(self.pending_grads)}")

                made_update = True
            else:
                # no hay suficiente quórum
                break

        # También, purgar gradientes demasiado viejos después de avanzar steps
        # (si alguno quedó fuera de bound)
        purged = 0
        new_deque = deque()
        for gmsg in self.pending_grads:
            if gmsg["step"] < self.global_step - self.ssp_bound:
                purged += 1
            else:
                new_deque.append(gmsg)
        if purged:
            self.total_batches_discarded += purged
            self.pending_grads = new_deque
            print(f"[PS][SSP] Purgados {purged} gradientes por sobrepasar bound tras avances.")

    def _is_training_finished(self) -> bool:
        """
        Termina cuando: no quedan épocas por despachar (self.epoch_idx >= epochs y train_iter agotado),
        todos los workers están libres, y no hay gradientes pendientes.
        """
        no_more_batches = (self.epoch_idx >= self.epochs)
        everyone_idle = all((not busy) for busy in self.worker_busy.values()) if self.worker_busy else False
        no_pending = len(self.pending_grads) == 0
        return no_more_batches and everyone_idle and no_pending

    def _finalize_and_stop(self):
        """Notifica STOP a todos y muestra resumen."""
        if self._t0_train is None:
            self._t0_train = time.perf_counter()
        self._t_end_train = time.perf_counter()

        # Enviar STOP
        for r, s in list(self.clients.items()):
            try:
                send_obj(s, {"type": "stop"})
            except Exception:
                pass

        # Resumen
        total_time = self._t_end_train - self._t0_train
        print("\n========== RESUMEN ==========")
        print(f"Epochs: {self.epochs}")
        print(f"Workers: {self.num_workers}")
        print(f"Quorum: {self.quorum} | SSP bound: {self.ssp_bound}")
        print(f"Batches enviados: {self.total_batches_sent}")
        print(f"Updates aplicados: {self.total_updates}")
        print(f"Gradientes descartados: {self.total_batches_discarded}")
        print(f"Tiempo total: {total_time:.3f}s")
        print("================================\n")


# ---------------------------
# Main
# ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ssp-bound", type=int, default=1,
                        help="Máxima distancia permitida entre el step global y el step de contribuciones elegibles.")
    parser.add_argument("--quorum", type=int, default=None,
                        help="Número mínimo de contribuciones por update. Por defecto=num_workers - ssp_bound (min 1)." )
    parser.add_argument("--loader-workers", type=int, default=2)
    parser.add_argument("--pin-memory", action="store_true")

    args = parser.parse_args()

    ps = ParameterServer(
        host=args.host,
        port=args.port,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ssp_bound=args.ssp_bound,
        quorum=args.quorum,
        num_workers_loader=args.loader_workers,
        pin_memory=args.pin_memory
    )
    ps.start()


if __name__ == "__main__":
    main()
