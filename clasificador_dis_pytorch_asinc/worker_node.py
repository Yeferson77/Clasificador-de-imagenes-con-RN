
import argparse
import io
import pickle
import socket
import struct
from typing import Any, Dict, List, Tuple

import psutil
import torch
import torch.nn as nn

import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler

import torchvision
import torchvision.transforms as T

torch.backends.cudnn.benchmark = True          # Acelera si el tamaño de input es estable
torch.set_float32_matmul_precision('high')


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
# Data loader con partición
# ---------------------------

def build_dataloader(rank: int, world_size: int, batch_size: int) -> Tuple[DataLoader, DistributedSampler]:
    """
    Crea DataLoader para CIFAR-10 (train) usando DistributedSampler para particionar
    el dataset entre workers (por rank/world_size). Cada epoch el sampler baraja de forma consistente.
    """
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.4914, 0.4822, 0.4465),
                    std=(0.2470, 0.2435, 0.2616)),
    ])

    train_set = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    loader = DataLoader(train_set, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True)
    return loader, sampler


# ---------------------------
# Worker principal
# ---------------------------

def run_worker(server_host: str, server_port: int, rank: int, world_size: int,
               batch_size: int, device_str: str = None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Worker {rank}] Usando dispositivo: {device}")

    # Conexión con el PS
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[Worker {rank}] Conectado a PS {server_host}:{server_port}")

    # Registro
    send_obj(sock, {"type": "register", "rank": rank, "world_size": world_size})
    cfg = recv_obj(sock)
    assert cfg["type"] == "config"

    ssp_bound = cfg.get("ssp_bound", None)
    quorum = cfg.get("quorum", None)
    if ssp_bound is not None:
        print(f"[Worker {rank}] Config SSP: bound={ssp_bound}, quorum={quorum}"), f"[Worker {rank}] Config esperada; recibido: {cfg.get('type')}"
    param_names = cfg["param_names"]
    epochs = cfg["epochs"]
    steps_per_epoch = cfg["steps_per_epoch"]
    lr = cfg["lr"]
    server_step = cfg["step"]

    # Modelo / criterio
    model = Cifar10CNN().to(device)
    # Carga de pesos iniciales desde el PS
    state_dict_cpu = cfg["state_dict"]
    model.load_state_dict({k: v.to(device) for k, v in state_dict_cpu.items()})
    criterion = nn.CrossEntropyLoss()

    # Data loader particionado
    train_loader, sampler = build_dataloader(rank, world_size, batch_size)
    data_iter = iter(train_loader)

    # Bucle de entrenamiento (sin optimizer local; solo computa gradientes)
    total_steps = epochs * steps_per_epoch
    local_steps_done = 0
    
    for epoch in range(epochs):
        # barajar de forma consistente el segmento de datos para este epoch
        sampler.set_epoch(epoch)
        data_iter = iter(train_loader)

        # Mostrar uso de RAM al inicio de la época
        ram_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"[Worker {rank}] Época {epoch+1}/{epochs} - Uso de RAM: {ram_mb:.2f} MB")

        for step_in_epoch in range(steps_per_epoch):
            # Obtener siguiente batch (si se agota, se reinicia el iterador)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

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

            # Enviar gradientes y esperar actualización
            send_obj(sock, {
                "type": "gradients",
                "worker": rank,
                "step": server_step,
                "batch_size": int(x.size(0)),
                "grads": grads_cpu
            })

            resp = recv_obj(sock)
            rtype = resp.get("type")

            if rtype == "update":
                # Cargar nuevos pesos y avanzar step
                state_cpu = resp["state_dict"]
                model.load_state_dict({k: v.to(device) for k, v in state_cpu.items()})
                server_step = resp["step"]
                local_steps_done += 1

                if (local_steps_done % 20 == 0) or (local_steps_done == total_steps):
                    print(f"[Worker {rank}] Progreso: {local_steps_done}/{total_steps} steps")

            elif rtype == "resync":
                # Re-sincronizar: actualizar pesos/step y repetir el step sin consumir batch extra
                state_cpu = resp["state_dict"]
                model.load_state_dict({k: v.to(device) for k, v in state_cpu.items()})
                server_step = resp["step"]
                print(f"[Worker {rank}] Resync recibido. Nuevo step={server_step}. Reintentando.")

                # Retroceder el contador del step_in_epoch para volver a intentar mismo step
                # (ajustamos con continue para repetir)
                continue

            elif rtype == "stop":
                print(f"[Worker {rank}] Entrenamiento finalizado por PS.")
                send_obj(sock, {"type": "done"})
                sock.close()
                return
            else:
                raise RuntimeError(f"[Worker {rank}] Respuesta desconocida del PS: {rtype}")

    # Si salimos del bucle natural (teórico), avisamos al PS
    print(f"[Worker {rank}] Entrenamiento local completado (pasos planificados).")
    try:
        send_obj(sock, {"type": "done"})
    except Exception:
        pass
    sock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-host", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=5000)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world-size", type=int,  default=3)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--device", type=str, default=None,
                        help="Forzar dispositivo, e.g., 'cuda' o 'cpu'. Por defecto: cuda si disponible.")
    args = parser.parse_args()

    run_worker(
        server_host=args.server_host,
        server_port=args.server_port,
        rank=args.rank,
        world_size=args.world_size,
        batch_size=args.batch_size,
        device_str=args.device
    )


if __name__ == "__main__":
    main()
    
    
# python worker_node.py --rank 1 
