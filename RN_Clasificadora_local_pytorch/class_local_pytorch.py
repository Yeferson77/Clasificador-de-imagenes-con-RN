#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clasificador ImageNet en PyTorch (versión LOCAL)
------------------------------------------------
- Carga ImageNet con torchvision (splits 'train' y 'val')
- Data augmentation estándar de ImageNet
- CNN tipo RN_Imagenet (como en ps_server_imagenet.py)
- Entrenamiento local (no distribuido)
- Mide tiempo por época y RAM
- Al final:
    * imprime en consola las mismas métricas de resumen que ps_server_imagenet.py
    * guarda un JSON con el resumen final en ./informes/
"""

import argparse
import time
import os
import json
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
import torchvision
import torchvision.transforms as T

# RAM opcional
try:
    import psutil
except Exception:
    psutil = None


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


def sample_ram_metrics():
    """
    Devuelve métricas de RAM similares a ps_server_imagenet:
    - ram_total_gb
    - ram_used_gb
    - ram_percent
    - proc_rss_gb
    """
    out = {
        "ram_total_gb": "N/A",
        "ram_used_gb": "N/A",
        "ram_percent": "N/A",
        "proc_rss_gb": "N/A",
    }
    if psutil is None:
        return out

    try:
        vm = psutil.virtual_memory()
        out["ram_total_gb"] = round(vm.total / (1024 ** 3), 3)
        out["ram_used_gb"] = round((vm.total - vm.available) / (1024 ** 3), 3)
        out["ram_percent"] = vm.percent
        proc = psutil.Process()
        rss = proc.memory_info().rss
        out["proc_rss_gb"] = round(rss / (1024 ** 3), 3)
    except Exception:
        pass

    return out


# ---------------------------
# Modelo (RN_Imagenet, igual que en el PS asíncrono)
# ---------------------------

class RN_Imagenet(nn.Module):
    """CNN simple ajustada para ImageNet (1k clases) usando AdaptiveAvgPool2d (entradas 224x224)."""
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
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56 -> 28

            nn.Dropout(0.5),
            nn.Conv2d(128, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28 -> 14
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
    parser.add_argument("--data", type=str, default="C:\Yeferson\Cursos\Cliente-Servidor\Dataset\data\imagenet",
                        help="Carpeta raíz de ImageNet (contiene subcarpetas 'train' y 'val')")
    parser.add_argument("--epochs", type=int, default=20, help="Épocas de entrenamiento")
    parser.add_argument("--batch_size", type=int, default=24, help="Tamaño de lote (224x224 es pesado)")
    parser.add_argument("--lr", type=float, default=4e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Proporción de validación tomada del split 'train' de ImageNet")
    parser.add_argument("--no_aug", action="store_true", help="Desactiva data augmentation fuerte (RandomResizedCrop)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_path", type=str, default="imagenet_local_cnn.pt",
                        help="Ruta para guardar el mejor modelo")
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # ---------------------------
    # Transforms para ImageNet
    # ---------------------------
    if args.no_aug:
        # Sin data augmentation fuerte
        train_tfms = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])
    else:
        train_tfms = T.Compose([
            T.RandomResizedCrop(224, scale=(0.08, 1.0), ratio=(3/4, 4/3)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406),
                        (0.229, 0.224, 0.225)),
        ])

    test_tfms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406),
                    (0.229, 0.224, 0.225)),
    ])

    # ---------------------------
    # Datasets: ImageNet
    # ---------------------------
    # train_full: split 'train'
    # test_ds: split 'val' (equivalente al "val" oficial de ILSVRC)
    train_full = torchvision.datasets.ImageNet(
        root=args.data,
        split="train",
        transform=train_tfms
    )
    test_ds = torchvision.datasets.ImageNet(
        root=args.data,
        split="val",
        transform=test_tfms
    )

    num_classes = len(getattr(train_full, "classes", [])) or 1000
    print(f"Número de clases ImageNet: {num_classes}")

    # Split entrenamiento/validación a partir de train_full
    val_size = int(len(train_full) * args.val_split)
    train_size = len(train_full) - val_size
    train_ds, val_ds = random_split(train_full, [train_size, val_size])

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Modelo + Optimizador + Criterio
    model = RN_Imagenet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []

    # Para métricas tipo ps_server_imagenet
    epoch_times = []
    epoch_accs_pct = []  # accuracy de validación por época (en %)
    epoch_ram = []

    print("\nInicio de entrenamiento (ImageNet)...\n")

    # Marca inicio de ENTRENAMIENTO (sin incluir evaluación final en test)
    t0_train = time.time()

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = run_one_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_one_epoch(
            model, val_loader, criterion, optimizer=None, device=device, train=False
        )

        scheduler.step()

        secs = time.time() - start
        history.append(EpochLog(epoch, train_loss, train_acc, val_loss, val_acc, secs))

        # Guardamos métricas para resumen tipo PS
        epoch_times.append(secs)
        epoch_accs_pct.append(val_acc * 100.0)  # pasar a %
        epoch_ram.append(sample_ram_metrics())

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

    # Marca fin de entrenamiento (antes de evaluar en test)
    t_end_train = time.time()
    total_training_time = t_end_train - t0_train

    # Evaluación en test con el mejor modelo si existe
    try:
        ckpt = torch.load(args.save_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"\nCargado mejor modelo de: {args.save_path} "
              f"(época={ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")
    except Exception as e:
        print(f"\nNo se pudo cargar el mejor modelo: {e}\nSe evalúa el modelo tal cual.")

    test_loss, test_acc = run_one_epoch(
        model, test_loader, criterion, optimizer=None, device=device, train=False
    )
    print(f"\n Test (split 'val' de ImageNet): loss={test_loss:.4f} acc={test_acc:.4f}")

    # Imprime tiempos por época (como antes)
    print("\nTiempos por época:")
    for log in history:
        print(f"Época {log.epoch:02d}: {log.secs:.2f} s")

    # ---------------------------
    # RESUMEN FINAL tipo ps_server_imagenet.py
    # ---------------------------
    # Global steps ≈ nº de steps de actualización (batches de entrenamiento)
    global_steps = args.epochs * len(train_loader)
    avg_step_time = total_training_time / max(global_steps, 1)
    final_val_accuracy_pct = test_acc * 100.0  # usamos accuracy en test como "final_val_accuracy"

    # Construimos epoch_metrics como en el PS
    epoch_metrics = []
    total_epochs = len(epoch_times)
    for i in range(total_epochs):
        tsec = epoch_times[i] if i < len(epoch_times) else None
        acc_ep = epoch_accs_pct[i] if i < len(epoch_accs_pct) else None
        ram = epoch_ram[i] if i < len(epoch_ram) else {}

        epoch_metrics.append({
            "epoch": i + 1,
            "time_seconds": tsec,
            "accuracy": acc_ep,
            "ram": ram,
        })

    # Imprimir en consola (mismas métricas que el PS asíncrono)
    print("\n========== RESUMEN FINAL ==========")
    print(f"Workers utilizados: 1 (entrenamiento local)")
    print(f"Épocas totales: {args.epochs}")
    print(f"Pasos (updates) globales: {global_steps}")
    print(f"Tiempo total de entrenamiento: {total_training_time:.3f} s")
    print(f"Tiempo promedio por update: {avg_step_time * 1000:.2f} ms")
    print(f"Accuracy final val (ImageNet): {final_val_accuracy_pct:.2f}%\n")

    print("---- Métricas por época ----")
    for m in epoch_metrics:
        tsec = m["time_seconds"]
        acc_ep = m["accuracy"]
        ram = m["ram"] or {}
        tsec_str = f"{tsec:.3f}s" if tsec is not None else "N/A"
        acc_str = f"{acc_ep:.2f}%" if (acc_ep is not None) else "N/A"
        ram_str = (
            f"RAM(sys): {ram.get('ram_used_gb', 'N/A')}/{ram.get('ram_total_gb', 'N/A')} GB "
            f"({ram.get('ram_percent', 'N/A')}%) | "
            f"RAM(proc): {ram.get('proc_rss_gb', 'N/A')} GB"
        ) if ram else "RAM: N/A"
        print(f"Época {m['epoch']:02d}: Tiempo = {tsec_str} | Accuracy = {acc_str} | {ram_str}")

    # Tasa de convergencia (Δacc entre épocas) como en el PS
    conv_rates = []
    for i in range(1, len(epoch_accs_pct)):
        prev, cur = epoch_accs_pct[i - 1], epoch_accs_pct[i]
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
    # Guardar resumen en JSON en ./informes
    # ---------------------------
    resumen_json = {
        "workers": 1,
        "epochs": args.epochs,
        "global_steps": global_steps,
        "total_training_time_seconds": total_training_time,
        "avg_step_time_seconds": avg_step_time,
        "final_val_accuracy": final_val_accuracy_pct,
        "epoch_metrics": epoch_metrics,
    }

    try:
        # Obtiene la carpeta del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))
        informes_dir = os.path.join(script_dir, "informes")
        os.makedirs(informes_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            informes_dir,
            f"reporte_final_imagenet_local_{timestamp}.json"
        )
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(resumen_json, f, ensure_ascii=False, indent=4)
        print(f"[LOCAL] Resumen final guardado en JSON: {json_path}")
    except Exception as e:
        print(f"[LOCAL] Error guardando resumen JSON en 'informes': {e}")


if __name__ == "__main__":
    main()
