"""Pregunta 3: arquitectura CNN en PyTorch y validación de shapes de tensores 4D."""
import torch
import torch.nn as nn


class CNN(nn.Module):
    """CNN para CIFAR-10: 3 bloques convolucionales + 2 capas densas.

    Flujo de tensores (B = batch):
        (B, 3, 32, 32)
        -> Bloque 1: Conv2d(3->32)   + BN + ReLU + MaxPool2d(2) -> (B, 32, 16, 16)
        -> Bloque 2: Conv2d(32->64)  + BN + ReLU + MaxPool2d(2) -> (B, 64,  8,  8)
        -> Bloque 3: Conv2d(64->128) + BN + ReLU + MaxPool2d(2) -> (B, 128, 4,  4)
        -> Flatten                                              -> (B, 2048)
        -> Linear(2048->256) + ReLU + Dropout(0.5)              -> (B, 256)
        -> Linear(256->10)                                      -> (B, 10)
    """

    def __init__(self, num_classes=10, in_channels=3, input_size=32, dropout=0.5):
        super().__init__()

        def conv_block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

        self.block1 = conv_block(in_channels, 32)
        self.block2 = conv_block(32, 64)
        self.block3 = conv_block(64, 128)

        # Cálculo dinámico del tamaño de entrada a la capa densa:
        # cada MaxPool2d(2) divide H y W a la mitad (3 bloques -> /8).
        feat_size = input_size // (2 ** 3)
        self.flat_dim = 128 * feat_size * feat_size   # 128 * 4 * 4 = 2048

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.flat_dim, 256)
        self.relu_fc = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        return self.fc2(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def validate_shapes(batch_size=8, verbose=True):
    """Pasa un lote simulado por el modelo, imprime los shapes intermedios y
    verifica con aserciones que los tensores 4D fluyen sin Shape Mismatch."""
    model = CNN()
    model.eval()
    x = torch.randn(batch_size, 3, 32, 32)   # lote simulado (B, C, H, W)
    assert x.dim() == 4, "La entrada debe ser un tensor 4D (B, C, H, W)"

    expected = [
        ("Entrada",                    (batch_size, 3, 32, 32)),
        ("Bloque 1 (Conv-ReLU-Pool)",  (batch_size, 32, 16, 16)),
        ("Bloque 2 (Conv-ReLU-Pool)",  (batch_size, 64, 8, 8)),
        ("Bloque 3 (Conv-ReLU-Pool)",  (batch_size, 128, 4, 4)),
        ("Flatten",                    (batch_size, 2048)),
        ("Linear 1 + ReLU",            (batch_size, 256)),
        ("Linear 2 (logits)",          (batch_size, 10)),
    ]
    steps = [
        model.block1, model.block2, model.block3, model.flatten,
        lambda t: model.relu_fc(model.fc1(t)), model.fc2,
    ]

    with torch.no_grad():
        shapes = [("Entrada", tuple(x.shape))]
        out = x
        for (name, _), step in zip(expected[1:], steps):
            out = step(out)
            shapes.append((name, tuple(out.shape)))

    if verbose:
        print("=" * 60)
        print("VALIDACIÓN DE SHAPES (forward pass con lote simulado)")
        print("=" * 60)
        for name, shape in shapes:
            print(f"{name:<28} -> {shape}")
        print(f"Parámetros entrenables: {count_parameters(model):,}")

    for (name, exp), (_, got) in zip(expected, shapes):
        assert got == exp, f"Shape Mismatch en '{name}': esperado {exp}, obtenido {got}"

    with torch.no_grad():
        full = model(x)   # forward() completo
    assert full.shape == (batch_size, 10), f"Salida inesperada: {tuple(full.shape)}"
    if verbose:
        print("OK: la arquitectura procesa tensores 4D sin errores de shape.")
        print("=" * 60)
    return model


if __name__ == "__main__":
    validate_shapes()
