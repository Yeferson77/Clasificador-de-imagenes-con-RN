"""
Ejecución (ejemplo con 2 workers):
    python worker_node.py --server-host <IP_DEL_PS> --server-port 5000 --rank 0 --world-size 2 --batch-size 128 --imagenet-root ./data/imagenet
    python worker_node.py --server-host <IP_DEL_PS> --server-port 5000 --rank 1 --world-size 2 --batch-size 128 --imagenet-root ./data/imagenet
"""

import argparse
import io
import pickle
import socket
import struct
from typing import Any, Dict, List, Tuple

import psutil
import torch
import torch.nn as nn

import torch.optim as optim  # no se usa directamente, pero se deja por compatibilidad
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

class RN_Imagenet(nn.Module):
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
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
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


# ---------------------------
# Data loader con partición (ImageNet)
# ---------------------------

def build_dataloader(rank: int, world_size: int, batch_size: int,
                     imagenet_root: str) -> Tuple[DataLoader, DistributedSampler]:
    """
    Crea DataLoader para ImageNet (train) usando DistributedSampler para particionar
    el dataset entre workers (por rank/world_size). Cada epoch el sampler baraja de forma consistente.
    """
    transform = T.Compose([
        T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
    ])

    train_set = torchvision.datasets.ImageNet(
        root=imagenet_root, split="train", transform=transform
    )
    sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank,
        shuffle=True, drop_last=True
    )
    loader = DataLoader(
        train_set, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True
    )
    return loader, sampler


# ---------------------------
# Worker principal
# ---------------------------

def run_worker(server_host: str, server_port: int, rank: int, world_size: int,
               batch_size: int, imagenet_root: str, device_str: str = None):
    device = torch.device(device_str if device_str else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[Worker {rank}] Usando dispositivo: {device}")

    # Conexión con el PS
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((server_host, server_port))
    print(f"[Worker {rank}] Conectado a PS {server_host}:{server_port}")

    # Registro
    send_obj(sock, {"type": "register", "rank": rank, "world_size": world_size})
    cfg = recv_obj(sock)
    assert cfg["type"] == "config", f"[Worker {rank}] Config esperada; recibido: {cfg.get('type')}"
    param_names = cfg["param_names"]
    epochs = cfg["epochs"]
    steps_per_epoch = cfg["steps_per_epoch"]
    lr = cfg["lr"]
    server_step = cfg["step"]
    num_classes = int(cfg.get("num_classes", 1000))

    # Modelo / criterio
    model = RN_Imagenet(num_classes=num_classes).to(device)
    # Carga de pesos iniciales desde el PS
    state_dict_cpu = cfg["state_dict"]
    model.load_state_dict({k: v.to(device) for k, v in state_dict_cpu.items()})
    criterion = nn.CrossEntropyLoss()

    # Data loader particionado (ImageNet train)
    train_loader, sampler = build_dataloader(rank, world_size, batch_size, imagenet_root)
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

                # Repetir el mismo step (no avanzamos step_in_epoch)
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
    parser.add_argument("--world-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--imagenet-root", type=str, default="../clasificador_dis_pytorch_asinc_imagenet/data/imagenet",
                        help="Ruta a la raíz que contiene los splits 'train' y 'val' de ILSVRC2012")
    parser.add_argument("--device", type=str, default=None,
                        help="Forzar dispositivo, e.g., 'cuda' o 'cpu'. Por defecto: cuda si disponible.")
    args = parser.parse_args()

    run_worker(
        server_host=args.server_host,
        server_port=args.server_port,
        rank=args.rank,
        world_size=args.world_size,
        batch_size=args.batch_size,
        imagenet_root=args.imagenet_root,
        device_str=args.device
    )


if __name__ == "__main__":
    main()

# comando manual ejemplo:
# python worker_node.py --server-host 127.0.0.1 --server-port 5000 --rank 0 --world-size 2 --batch-size 128 --imagenet-root ./data/imagenet
