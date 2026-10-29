"""Pregunta 4: entrenamiento, métricas, curvas de aprendizaje y matriz de confusión."""
import json
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from dataset import SEED, ROOT, OUT_DIR, CLASSES, set_seed, get_dataloaders
from model import CNN, count_parameters, validate_shapes

# ----------------------------------------------------------------------------
# Hiperparámetros
# ----------------------------------------------------------------------------
EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_epoch(model, loader, criterion, optimizer=None):
    """Una época completa. Entrena si se pasa optimizer; evalúa en caso contrario."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(dim=1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, labels = [], []
    for x, y in loader:
        out = model(x.to(DEVICE))
        preds.append(out.argmax(dim=1).cpu())
        labels.append(y)
    return torch.cat(preds).numpy(), torch.cat(labels).numpy()


def plot_curves(history, path):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].plot(epochs, history["train_loss"], "o-", label="Entrenamiento")
    ax[0].plot(epochs, history["val_loss"], "s-", label="Validación")
    ax[0].set_title("Pérdida (CrossEntropyLoss)")
    ax[0].set_xlabel("Época"); ax[0].set_ylabel("Loss"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(epochs, [a * 100 for a in history["train_acc"]], "o-", label="Entrenamiento")
    ax[1].plot(epochs, [a * 100 for a in history["val_acc"]], "s-", label="Validación")
    ax[1].set_title("Accuracy (%)")
    ax[1].set_xlabel("Época"); ax[1].set_ylabel("Accuracy (%)"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle(f"Curvas de aprendizaje - CNN CIFAR-10 (semilla {SEED})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(cm, path, test_acc):
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(10)); ax.set_yticks(range(10))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Clase predicha"); ax.set_ylabel("Clase real")
    ax.set_title(f"Matriz de confusión - conjunto de prueba (accuracy {test_acc * 100:.2f}%)")
    thr = cm.max() / 2
    for i in range(10):
        for j in range(10):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > thr else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_analysis(h, test_acc, per_class_acc):
    """Análisis cuantitativo de overfitting / underfitting con las métricas reales."""
    tr_acc, va_acc = h["train_acc"][-1] * 100, h["val_acc"][-1] * 100
    tr_loss, va_loss = h["train_loss"][-1], h["val_loss"][-1]
    gap = tr_acc - va_acc
    best_ep = int(np.argmin(h["val_loss"])) + 1
    min_val_loss = min(h["val_loss"])
    loss_rise = va_loss - min_val_loss
    n_ep = len(h["train_loss"])

    if gap > 8 or (gap > 4 and loss_rise > 0.10):
        diag = "**Sobreajuste (overfitting)**"
        expl = (f"La brecha train-val es de {gap:.2f} puntos porcentuales y la pérdida de validación "
                f"está {loss_rise:.3f} por encima de su mínimo (época {best_ep}). El modelo empieza a "
                "memorizar el conjunto de entrenamiento y generaliza peor.")
    elif tr_acc < 70 and va_acc < 70:
        diag = "**Subajuste (underfitting)**"
        expl = (f"Tanto el accuracy de entrenamiento ({tr_acc:.2f}%) como el de validación ({va_acc:.2f}%) "
                "son bajos: el modelo no logra capturar la complejidad del problema y requiere más "
                "capacidad, más épocas o un mejor ajuste del learning rate.")
    else:
        diag = "**Ajuste adecuado (sin overfitting ni underfitting relevantes)**"
        if gap < 0:
            extra = ("El accuracy de validación es incluso mayor que el de entrenamiento porque el "
                     "entrenamiento se mide con Data Augmentation y Dropout activos, que dificultan "
                     "las muestras; la validación se evalúa sin ellos.")
        else:
            extra = ("La brecha es pequeña, lo que indica que la regularización (Data Augmentation, "
                     "BatchNorm y Dropout) controla bien la memorización.")
        expl = (f"La brecha train-val es de {gap:.2f} puntos porcentuales y la pérdida de validación "
                f"se mantiene {loss_rise:.3f} sobre su mínimo (época {best_ep}). {extra}")

    best_cls = int(np.argmax(per_class_acc)); worst_cls = int(np.argmin(per_class_acc))
    return (
        f"**Resultados con la semilla {SEED}** ({n_ep} épocas, Adam lr={LEARNING_RATE}, batch={BATCH_SIZE}, "
        f"dispositivo: {DEVICE.type}):\n\n"
        "| Métrica | Entrenamiento | Validación |\n|---|---|---|\n"
        f"| Loss final | {tr_loss:.4f} | {va_loss:.4f} |\n"
        f"| Accuracy final | {tr_acc:.2f}% | {va_acc:.2f}% |\n\n"
        f"- Mejor época por pérdida de validación: **{best_ep}** (val loss = {min_val_loss:.4f}).\n"
        f"- Accuracy en el conjunto de prueba independiente (mejor modelo por val accuracy): **{test_acc * 100:.2f}%**.\n"
        f"- Clase mejor clasificada: **{CLASSES[best_cls]}** ({per_class_acc[best_cls] * 100:.1f}%). "
        f"Clase más difícil: **{CLASSES[worst_cls]}** ({per_class_acc[worst_cls] * 100:.1f}%).\n\n"
        f"**Diagnóstico:** {diag}. {expl}\n\n"
        "![Curvas de aprendizaje](outputs/loss_accuracy.png)\n\n"
        "![Matriz de confusión](outputs/confusion_matrix.png)"
    )


def update_readme(section_md):
    """Inserta el análisis entre los marcadores del README (raíz y outputs/)."""
    readme = ROOT / "README.md"
    if not readme.exists():
        print("README.md no encontrado: se omite la actualización.")
        return
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(r"(<!-- RESULTADOS:START -->).*?(<!-- RESULTADOS:END -->)", re.S)
    new_text = pattern.sub(lambda m: f"{m.group(1)}\n{section_md}\n{m.group(2)}", text)
    readme.write_text(new_text, encoding="utf-8")
    # Copia en outputs/ (la guía la ubica ahí) con rutas de imagen relativas a esa carpeta
    (OUT_DIR / "README.md").write_text(new_text.replace("](outputs/", "]("), encoding="utf-8")
    print("README.md actualizado con las métricas reales (raíz y outputs/).")


def main():
    set_seed(SEED)
    OUT_DIR.mkdir(exist_ok=True)
    print(f"Semilla: {SEED} | Dispositivo: {DEVICE}")

    validate_shapes()   # evidencia de la Pregunta 3 antes de entrenar

    train_loader, val_loader, test_loader = get_dataloaders(BATCH_SIZE)

    set_seed(SEED)      # pesos iniciales reproducibles
    model = CNN().to(DEVICE)
    print(model)
    print(f"Parámetros entrenables: {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc, best_state = 0.0, None

    for epoch in range(EPOCHS):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion)
        history["train_loss"].append(tr_loss); history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss);   history["val_acc"].append(va_acc)
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Época {epoch + 1:02d}/{EPOCHS} | train loss {tr_loss:.4f} acc {tr_acc * 100:.2f}% | "
              f"val loss {va_loss:.4f} acc {va_acc * 100:.2f}% | {time.time() - t0:.1f}s")

    # Evaluación final en el conjunto de prueba con el mejor modelo (por val accuracy)
    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / f"cnn_cifar10_seed{SEED}.pt")
    preds, labels = predict(model, test_loader)
    test_acc = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)

    print(f"\nAccuracy en prueba: {test_acc * 100:.2f}%")
    print(classification_report(labels, preds, target_names=CLASSES, digits=4))

    plot_curves(history, OUT_DIR / "loss_accuracy.png")
    plot_confusion(cm, OUT_DIR / "confusion_matrix.png", test_acc)
    with open(OUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SEED, "epochs": EPOCHS, "test_accuracy": test_acc,
                   "history": history,
                   "per_class_accuracy": {c: float(a) for c, a in zip(CLASSES, per_class_acc)}},
                  f, indent=2, ensure_ascii=False)
    print("Guardados: outputs/loss_accuracy.png, outputs/confusion_matrix.png, outputs/metrics.json")

    update_readme(build_analysis(history, test_acc, per_class_acc))


if __name__ == "__main__":
    main()
