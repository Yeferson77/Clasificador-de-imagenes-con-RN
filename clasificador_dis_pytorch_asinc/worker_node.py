import argparse
import pickle
import socket
import struct
from typing import Any, Dict, List

import torch
import torch.nn as nn

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
# Modelo CNN (debe coincidir con el PS)
# ---------------------------

class Cifar10CNN(nn.Module):
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


# ---------------------------
# Worker principal
# ---------------------------

def run_worker(server_host: str, server_port: int, rank: int, world_size: int, device_str: str = None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Worker {rank}] Dispositivo: {device}")

    # Conexión con el PS
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[Worker {rank}] Conectado a PS {server_host}:{server_port}")

    # Registro
    send_obj(sock, {"type": "register", "rank": rank, "world_size": world_size})
    cfg = recv_obj(sock)
    assert cfg["type"] == "config"

    param_names = cfg["param_names"]
    server_step = cfg["step"]

    # Modelo / criterio
    model = Cifar10CNN().to(device)
    state_dict_cpu = cfg["state_dict"]
    model.load_state_dict({k: v.to(device) for k, v in state_dict_cpu.items()})
    criterion = nn.CrossEntropyLoss()

    # Tras registrar, indicamos al PS que estamos listos
    send_obj(sock, {"type": "ready", "rank": rank})

    try:
        while True:
            msg = recv_obj(sock)
            mtype = msg.get("type")

            if mtype == "work":
                # Cargar pesos enviados para este trabajo y step asignado
                state_cpu = msg["state_dict"]
                model.load_state_dict({k: v.to(device) for k, v in state_cpu.items()})
                server_step = msg["step"]

                x = msg["x"].to(device, non_blocking=True)
                y = msg["y"].to(device, non_blocking=True)

                # Compute gradientes
                model.train()
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.zero_()

                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()

                # Extraer gradientes en el ORDEN de param_names y mover a CPU para serializar
                grads_cpu: List[torch.Tensor] = []
                name_to_param = dict(model.named_parameters())
                for name in param_names:
                    g = name_to_param[name].grad
                    if g is None:
                        grads_cpu.append(torch.zeros_like(name_to_param[name].data, device="cpu"))
                    else:
                        grads_cpu.append(g.detach().to("cpu"))

                # Enviar gradientes y declararse listo para más trabajo
                send_obj(sock, {
                    "type": "gradients",
                    "worker": rank,
                    "step": server_step,
                    "batch_size": int(x.size(0)),
                    "grads": grads_cpu
                })

                send_obj(sock, {"type": "ready", "rank": rank})

            elif mtype == "stop":
                print(f"[Worker {rank}] STOP recibido. Saliendo.")
                send_obj(sock, {"type": "done"})
                break

            else:
                raise RuntimeError(f"[Worker {rank}] Mensaje desconocido del PS: {mtype}")

    finally:
        try:
            sock.close()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
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
