"""Pregunta 1: carga, visualización, transformaciones y partición estratificada de CIFAR-10."""
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------------
# Semilla individual: últimos 2 dígitos del código de matrícula
# ----------------------------------------------------------------------------
SEED = 75

ROOT = Path(__file__).resolve().parents[1]   # raíz del repositorio
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"

CLASSES = ("avión", "automóvil", "pájaro", "gato", "ciervo",
           "perro", "rana", "caballo", "barco", "camión")
MEAN = (0.4914, 0.4822, 0.4465)   # media por canal RGB de CIFAR-10 (train)
STD = (0.2470, 0.2435, 0.2616)    # desviación estándar por canal RGB


def set_seed(seed=SEED):
    """Fija todas las fuentes de aleatoriedad con la semilla individual."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_transforms():
    """Pipeline de transformaciones.

    - Entrenamiento: RandomCrop + RandomHorizontalFlip (data augmentation),
      ToTensor y Normalize por canal RGB.
    - Validación / prueba: solo ToTensor y Normalize (sin aumento).
    """
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, eval_tf


def get_datasets(root=DATA_DIR, val_size=0.1):
    """Descarga CIFAR-10 y realiza la partición estratificada train/val."""
    set_seed(SEED)
    train_tf, eval_tf = get_transforms()

    full_train_aug = datasets.CIFAR10(root, train=True, download=True, transform=train_tf)
    full_train_eval = datasets.CIFAR10(root, train=True, download=False, transform=eval_tf)
    test_set = datasets.CIFAR10(root, train=False, download=True, transform=eval_tf)

    # Partición estratificada (misma proporción de clases en train y val)
    idx = np.arange(len(full_train_aug))
    targets = np.array(full_train_aug.targets)
    train_idx, val_idx = train_test_split(
        idx, test_size=val_size, stratify=targets, random_state=SEED
    )
    train_set = Subset(full_train_aug, train_idx)   # con data augmentation
    val_set = Subset(full_train_eval, val_idx)      # sin data augmentation

    print(f"Semilla individual: {SEED}")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")
    print(f"Shape de una imagen (C, H, W): {tuple(train_set[0][0].shape)}")
    return train_set, val_set, test_set


def get_dataloaders(batch_size=64, root=DATA_DIR):
    train_set, val_set, test_set = get_datasets(root)
    g = torch.Generator().manual_seed(SEED)   # mezcla (shuffle) reproducible
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def show_samples(root=DATA_DIR, n=16, save_path=OUT_DIR / "samples_grid.png", show=True):
    """Cuadrícula de n imágenes (sin normalizar) con su etiqueta real."""
    set_seed(SEED)
    raw = datasets.CIFAR10(root, train=True, download=True,
                           transform=transforms.ToTensor())
    idxs = np.random.choice(len(raw), n, replace=False)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(8, 2.2 * rows))
    for ax, i in zip(axes.flat, idxs):
        img, label = raw[i]
        ax.imshow(img.permute(1, 2, 0))   # CHW -> HWC para matplotlib
        ax.set_title(f"{CLASSES[label]}", fontsize=10)
        ax.axis("off")
    fig.suptitle(f"Muestras de CIFAR-10 con su etiqueta real (semilla {SEED})")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    print(f"Cuadrícula guardada en: {save_path}")


if __name__ == "__main__":
    show_samples(show=False)
    get_dataloaders()
