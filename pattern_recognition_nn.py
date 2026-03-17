import itertools
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


# 5x7 bitmap templates for digits 0-9
DIGIT_TEMPLATES = {
    0: [
        "01110",
        "10001",
        "10011",
        "10101",
        "11001",
        "10001",
        "01110",
    ],
    1: [
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    2: [
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    3: [
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    4: [
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    5: [
        "11111",
        "10000",
        "10000",
        "11110",
        "00001",
        "00001",
        "11110",
    ],
    6: [
        "01110",
        "10000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    7: [
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    8: [
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    9: [
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00001",
        "01110",
    ],
}


def parse_templates() -> np.ndarray:
    rows = []
    for digit in range(10):
        bitmap = DIGIT_TEMPLATES[digit]
        flat = [int(pixel) for line in bitmap for pixel in line]
        rows.append(flat)
    return np.array(rows, dtype=np.float64)


def one_hot(indexes: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((indexes.shape[0], n_classes), dtype=np.float64)
    out[np.arange(indexes.shape[0]), indexes] = 1.0
    return out


def make_dataset(
    samples_per_class: int = 25,
    noise_probability: float = 0.08,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    templates = parse_templates()

    x_samples, y_samples = [], []
    for cls, template in enumerate(templates):
        for _ in range(samples_per_class):
            noisy = template.copy()
            flips = rng.random(noisy.shape[0]) < noise_probability
            noisy[flips] = 1.0 - noisy[flips]
            x_samples.append(noisy)
            y_samples.append(cls)

    x = np.array(x_samples, dtype=np.float64)
    y = np.array(y_samples, dtype=np.int32)

    indices = rng.permutation(x.shape[0])
    x = x[indices]
    y = y[indices]

    split_idx = int(train_ratio * x.shape[0])
    x_train, x_test = x[:split_idx], x[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return x_train, one_hot(y_train, 10), x_test, y_test


@dataclass
class Activation:
    forward: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]


ACTIVATIONS = {
    "sigmoid": Activation(
        forward=lambda z: 1.0 / (1.0 + np.exp(-z)),
        derivative=lambda z, a: a * (1.0 - a),
    ),
    "tanh": Activation(
        forward=np.tanh,
        derivative=lambda z, a: 1.0 - a**2,
    ),
    "arctan": Activation(
        forward=np.arctan,
        derivative=lambda z, a: 1.0 / (1.0 + z**2),
    ),
}


class MLP:
    def __init__(self, layer_sizes: List[int], activation_name: str, learning_rate: float, seed: int = 0):
        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation_name}")
        self.layer_sizes = layer_sizes
        self.activation = ACTIVATIONS[activation_name]
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.weights.append(self.rng.uniform(-limit, limit, size=(in_dim, out_dim)))
            self.biases.append(np.zeros((1, out_dim), dtype=np.float64))

    def forward(self, x: np.ndarray):
        activations = [x]
        zs = []
        a = x
        for w, b in zip(self.weights, self.biases):
            z = a @ w + b
            a = self.activation.forward(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def predict(self, x: np.ndarray) -> np.ndarray:
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.activation.forward(a @ w + b)
        return np.argmax(a, axis=1)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000):
        n = x.shape[0]
        for _ in range(epochs):
            activations, zs = self.forward(x)

            delta = (activations[-1] - y) * self.activation.derivative(zs[-1], activations[-1])
            grad_w = [None] * len(self.weights)
            grad_b = [None] * len(self.biases)

            grad_w[-1] = activations[-2].T @ delta / n
            grad_b[-1] = np.mean(delta, axis=0, keepdims=True)

            for l in range(2, len(self.layer_sizes)):
                z = zs[-l]
                a = activations[-l]
                delta = (delta @ self.weights[-l + 1].T) * self.activation.derivative(z, a)
                grad_w[-l] = activations[-l - 1].T @ delta / n
                grad_b[-l] = np.mean(delta, axis=0, keepdims=True)

            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * grad_w[i]
                self.biases[i] -= self.learning_rate * grad_b[i]


def run_experiments():
    x_train, y_train, x_test, y_test = make_dataset()

    hidden_variants = {
        0: [],
        1: [24],
        2: [32, 16],
    }
    learning_rates = [0.01, 0.05, 0.1]
    activation_names = ["sigmoid", "tanh", "arctan"]
    epochs = 1000

    results = []
    for activation_name, hidden_layers, lr in itertools.product(
        activation_names,
        sorted(hidden_variants.keys()),
        learning_rates,
    ):
        architecture = [x_train.shape[1], *hidden_variants[hidden_layers], y_train.shape[1]]
        model = MLP(architecture, activation_name=activation_name, learning_rate=lr, seed=7)
        model.fit(x_train, y_train, epochs=epochs)
        pred = model.predict(x_test)
        acc = float(np.mean(pred == y_test))
        results.append(
            {
                "activation": activation_name,
                "hidden_layers": hidden_layers,
                "architecture": architecture,
                "learning_rate": lr,
                "accuracy": acc,
            }
        )

    results.sort(key=lambda r: r["accuracy"], reverse=True)

    print("Результаты экспериментов (по убыванию точности):")
    print("-" * 88)
    print(f"{'Activation':<12} {'Hidden':<6} {'LearningRate':<12} {'Architecture':<18} {'Accuracy':<8}")
    print("-" * 88)
    for r in results:
        print(
            f"{r['activation']:<12} {r['hidden_layers']:<6} {r['learning_rate']:<12.2f} "
            f"{str(r['architecture']):<18} {r['accuracy']:.4f}"
        )

    best = results[0]
    print("\nОптимальные параметры сети:")
    print(f"- Функция активации: {best['activation']}")
    print(f"- Количество скрытых слоёв: {best['hidden_layers']}")
    print(f"- Архитектура: {best['architecture']}")
    print(f"- Норма обучения: {best['learning_rate']}")
    print(f"- Точность на тестовой выборке: {best['accuracy']:.4f}")


if __name__ == "__main__":
    run_experiments()
