"""
Ejecución (ejemplo con 2 workers):
    python worker_node_imagenet.py --server-host 127.0.0.1 --server-port 5000 --rank 0 --world-size 2
    python worker_node_imagenet.py --server-host 127.0.0.1 --server-port 5000 --rank 1 --world-size 2
"""
import argparse
import pickle
import socket
import struct
from typing import Any, Dict, List

import psutil
import torch
import torch.nn as nn

torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass


# ---------------------------
# Utilidades de red (sockets)
# ---------------------------

def send_obj(sock: socket.socket, obj: Any) -> None:
    data = pickle.dumps(obj, protocol=4)
    sock.sendall(struct.pack("!Q", len(data)))
    sock.sendall(data)


def recvall(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Conexión cerrada por el servidor")
        buf.extend(chunk)
    return bytes(buf)


def recv_obj(sock: socket.socket) -> Any:
    raw_len = recvall(sock, 8)
    (length,) = struct.unpack("!Q", raw_len)
    data = recvall(sock, length)
    return pickle.loads(data)


# ---------------------------
# Modelo CNN (igual al PS)
# ---------------------------

class Cifar10CNN(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Dropout(0.5),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ---------------------------
# Worker principal (SSP)
# ---------------------------

def run_worker(server_host: str, server_port: int, rank: int, world_size: int, device_str: str = None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Worker {rank}] Usando dispositivo: {device}")

    # Conexión con el PS
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[Worker {rank}] Conectado a PS {server_host}:{server_port}")

    # Registro y config inicial
    send_obj(sock, {"type": "register", "rank": rank, "world_size": world_size})
    cfg = recv_obj(sock)
    assert cfg["type"] == "config", f"[Worker {rank}] Se esperaba 'config', recibido: {cfg.get('type')}"
    param_names: List[str] = cfg["param_names"]
    epochs = cfg["epochs"]
    lr = cfg["lr"]
    batch_size = cfg["batch_size"]
    quorum = cfg["quorum"]
    ssp_bound = cfg["ssp_bound"]
    num_classes = int(cfg.get("num_classes", 1000))

    server_step = int(cfg["step"])

    # Modelo / criterio
    model = Cifar10CNN(num_classes=num_classes).to(device)
    state_dict_cpu = cfg["state_dict"]
    model.load_state_dict({k: v.to(device) for k, v in state_dict_cpu.items()})
    criterion = nn.CrossEntropyLoss()

    # Contadores
    local_steps_done = 0
    print(f"[Worker {rank}] SSP-bound={ssp_bound}, quorum={quorum}, batch_size={batch_size}, num_classes={num_classes}")

    # Bucle principal: recibir lote -> calcular gradientes -> enviar -> esperar siguiente instrucción
    while True:
        msg = recv_obj(sock)
        mtype = msg.get("type")

        if mtype == "train":
            # Posible actualización de pesos (piggyback)
            new_state = msg.get("state_dict")
            if new_state is not None:
                model.load_state_dict({k: v.to(device) for k, v in new_state.items()})
                server_step = int(msg.get("step", server_step))
            else:
                server_step = int(msg.get("step", server_step))

            epoch = int(msg["epoch"])
            batch_id = int(msg["batch_id"])
            x = msg["data"].to(device, non_blocking=True)
            y = msg["targets"].to(device, non_blocking=True)

            # Métrica de RAM
            if (local_steps_done % 20) == 0:
                ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                print(f"[Worker {rank}] Época {epoch} - RAM: {ram_mb:.2f} MB")

            # Entrenamiento local - solo gradientes
            model.train()
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # Extraer gradientes en el MISMO ORDEN del PS
            grads_cpu: List[torch.Tensor] = []
            name_to_param = dict(model.named_parameters())
            for name in param_names:
                g = name_to_param[name].grad
                grads_cpu.append(torch.zeros_like(name_to_param[name].data, device="cpu") if g is None else g.detach().to("cpu"))

            # Enviar gradientes (con la versión de modelo que estamos usando, para SSP)
            send_obj(sock, {
                "type": "gradients",
                "worker": rank,
                "model_step_used": int(server_step),
                "batch_size": int(x.size(0)),
                "grads": grads_cpu
            })
            local_steps_done += 1

        elif mtype == "stop":
            print(f"[Worker {rank}] Entrenamiento finalizado por PS. Steps locales: {local_steps_done}")
            try:
                send_obj(sock, {"type": "done"})
            except Exception:
                pass
            sock.close()
            return

        else:
            raise RuntimeError(f"[Worker {rank}] Mensaje desconocido del PS: {mtype}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--device", type=str, default=None,
                        help="Forzar dispositivo, e.g., 'cuda' o 'cpu'. Por defecto: cuda si disponible.")
    args = parser.parse_args()

    run_worker(
        server_host=args.server_host,
        server_port=args.server_port,
        rank=args.rank,
        world_size=args.world_size,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
