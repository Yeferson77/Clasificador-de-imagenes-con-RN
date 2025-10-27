
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificador CIFAR-10 en PyTorch (versión local)
------------------------------------------------
- Carga CIFAR-10 con torchvision
- Data augmentation (opcional)
- CNN sencilla con BatchNorm y Dropout
- Entrenamiento con temporizador por época
- Evalúa accuracy en validación/prueba
- Guarda el mejor modelo (según accuracy de validación)
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T


# ---------------------------
# Utilidades
# ---------------------------

def set_seed(seed: int = 42):
    import random
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


@dataclass
class EpochLog:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    secs: float


# ---------------------------
# Modelo
# ---------------------------

class SmallCIFAR10CNN(nn.Module):
    """
    CNN para CIFAR-10.
    
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 16x16
            nn.Dropout(0.1),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 8x8
            nn.Dropout(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # 4x4
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ---------------------------
# Entrenamiento / Evaluación
# ---------------------------

def accuracy_from_logits(logits, targets):
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def run_one_epoch(model, loader, criterion, optimizer, device, train: bool = True):
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_acc = 0.0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            acc = accuracy_from_logits(logits, labels)

            if train:
                loss.backward()
                optimizer.step()

        bs = images.size(0)
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_count += bs

    return total_loss / total_count, total_acc / total_count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="./data", help="Carpeta para CIFAR-10")
    parser.add_argument("--epochs", type=int, default=250, help="Épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=128, help="Tamaño de lote")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.1, help="Proporción de validación del set de entrenamiento")
    parser.add_argument("--no_aug", action="store_true", help="Desactiva data augmentation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_path", type=str, default="cifar10_cnn.pt", help="Ruta para guardar mejor modelo")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # Transforms
    if args.no_aug:
        train_tfms = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        train_tfms = T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    test_tfms = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Datasets
    train_full = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_tfms)
    test_ds = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=test_tfms)

    # Split entrenamiento/validación
    val_size = int(len(train_full) * args.val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # Modelo + Optimizador + Criterio
    model = SmallCIFAR10CNN(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []

    print("\nInicio de entrenamiento...\n")
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_one_epoch(model, val_loader, criterion, optimizer, device, train=False)

        scheduler.step()

        secs = time.time() - start
        history.append(EpochLog(epoch, train_loss, train_acc, val_loss, val_acc, secs))

        print(f" Época {epoch:02d} | {secs:6.2f}s | "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc": val_acc,
                "args": vars(args),
            }, args.save_path)
            print(f" Mejor modelo actualizado (val_acc={val_acc:.4f}) -> {args.save_path}")

    # Evaluación en test con el mejor modelo si existe
    try:
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nCargado mejor modelo de: {args.save_path} (época={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    except Exception as e:
        print(f"\nNo se pudo cargar el mejor modelo: {e}\nSe evalúa el modelo tal cual.")

    test_loss, test_acc = run_one_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    print(f"\n Test: loss={test_loss:.4f} acc={test_acc:.4f}")

    # Imprime tiempos por época
    print("\nTiempos por época:")
    for log in history:
        print(f"Época {log.epoch:02d}: {log.secs:.2f} s")

    # Muestra clases
    print("\nClases CIFAR-10:", train_full.classes)


if __name__ == "__main__":
    main()