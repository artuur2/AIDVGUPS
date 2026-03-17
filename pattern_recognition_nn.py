import gzip
import random
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk


@dataclass
class Activation:
    forward: Callable[[np.ndarray], np.ndarray]
    derivative: Callable[[np.ndarray, np.ndarray], np.ndarray]


ACTIVATIONS: Dict[str, Activation] = {
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
    def __init__(self, layer_sizes: Sequence[int], activation_name: str, learning_rate: float, seed: int = 0):
        if activation_name not in ACTIVATIONS:
            raise ValueError(f"Неизвестная функция активации: {activation_name}")
        self.layer_sizes = list(layer_sizes)
        self.activation = ACTIVATIONS[activation_name]
        self.learning_rate = learning_rate
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        for in_dim, out_dim in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.weights.append(self.rng.uniform(-limit, limit, size=(in_dim, out_dim)))
            self.biases.append(np.zeros((1, out_dim), dtype=np.float64))

    def _forward(self, x: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [x]
        zs = []
        a = x
        for w, b in zip(self.weights, self.biases):
            z = a @ w + b
            a = self.activation.forward(z)
            zs.append(z)
            activations.append(a)
        return activations, zs

    def _backward_update(self, x: np.ndarray, y: np.ndarray) -> None:
        n = x.shape[0]
        activations, zs = self._forward(x)
        delta = (activations[-1] - y) * self.activation.derivative(zs[-1], activations[-1])
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)
        grad_w[-1] = activations[-2].T @ delta / n
        grad_b[-1] = np.mean(delta, axis=0, keepdims=True)

        for l in range(2, len(self.layer_sizes)):
            delta = (delta @ self.weights[-l + 1].T) * self.activation.derivative(zs[-l], activations[-l])
            grad_w[-l] = activations[-l - 1].T @ delta / n
            grad_b[-l] = np.mean(delta, axis=0, keepdims=True)

        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 1000, batch_size: int = 128) -> None:
        n = x.shape[0]
        if batch_size <= 0:
            batch_size = n

        for _ in range(epochs):
            perm = self.rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                self._backward_update(x[idx], y[idx])

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.activation.forward(a @ w + b)
        return a

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_logits(x), axis=1)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), n_classes), dtype=np.float64)
    out[np.arange(len(y)), y] = 1.0
    return out


def pil_to_vector(img: Image.Image, image_size: int) -> np.ndarray:
    prepared = ImageOps.fit(img.convert("L"), (image_size, image_size))
    arr = np.asarray(prepared, dtype=np.float64) / 255.0
    return arr.reshape(1, -1)


def _open_binary(path: Path):
    return gzip.open(path, "rb") if path.suffix == ".gz" else open(path, "rb")


def read_idx_images(path: Path) -> np.ndarray:
    with _open_binary(path) as f:
        magic, count, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Некорректный файл изображений IDX: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count, rows * cols).astype(np.float64) / 255.0


def read_idx_labels(path: Path) -> np.ndarray:
    with _open_binary(path) as f:
        magic, count = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Некорректный файл меток IDX: {path}")
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(count).astype(np.int32)


def _find_mnist_file(folder: Path, names: Sequence[str]) -> Path:
    for name in names:
        path = folder / name
        if path.exists():
            return path
    raise ValueError(f"Не найден один из файлов: {', '.join(names)}")


def load_mnist_dataset(folder: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images_file = _find_mnist_file(folder, ["train-images-idx3-ubyte", "train-images-idx3-ubyte.gz"])
    train_labels_file = _find_mnist_file(folder, ["train-labels-idx1-ubyte", "train-labels-idx1-ubyte.gz"])
    test_images_file = _find_mnist_file(folder, ["t10k-images-idx3-ubyte", "t10k-images-idx3-ubyte.gz"])
    test_labels_file = _find_mnist_file(folder, ["t10k-labels-idx1-ubyte", "t10k-labels-idx1-ubyte.gz"])

    x_train = read_idx_images(train_images_file)
    y_train = read_idx_labels(train_labels_file)
    x_test = read_idx_images(test_images_file)
    y_test = read_idx_labels(test_labels_file)

    if len(x_train) != len(y_train) or len(x_test) != len(y_test):
        raise ValueError("Размеры изображений и меток MNIST не совпадают")

    return x_train, y_train, x_test, y_test


def load_dataset_from_folder(folder: Path, image_size: int) -> Tuple[np.ndarray, np.ndarray, List[str], List[Path]]:
    if not folder.exists() or not folder.is_dir():
        raise ValueError("Папка с эталонами не найдена")

    class_dirs = sorted([d for d in folder.iterdir() if d.is_dir()])
    if not class_dirs:
        raise ValueError("В папке эталонов нет подпапок классов")

    x_data: List[np.ndarray] = []
    y_data: List[int] = []
    labels: List[str] = []
    all_files: List[Path] = []

    for class_idx, class_dir in enumerate(class_dirs):
        labels.append(class_dir.name)
        image_files = sorted([p for p in class_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}])
        for img_path in image_files:
            img = Image.open(img_path)
            x_data.append(pil_to_vector(img, image_size))
            y_data.append(class_idx)
            all_files.append(img_path)

    if not x_data:
        raise ValueError("Эталонные изображения не найдены")

    return np.vstack(x_data), np.array(y_data, dtype=np.int32), labels, all_files


def image_to_vector(path: Path, image_size: int) -> np.ndarray:
    return pil_to_vector(Image.open(path), image_size)


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Распознавание арабских цифр (MLP)")
        self.root.geometry("1120x760")

        self.train_dir: Optional[Path] = None
        self.random_dir: Optional[Path] = None
        self.mnist_dir: Optional[Path] = None
        self.labels: List[str] = []
        self.model: Optional[MLP] = None
        self.image_size = tk.IntVar(value=28)

        self.canvas_size = 280
        self.brush_size = 16
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_ctx = ImageDraw.Draw(self.draw_image)

        self._build_ui()

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        settings = ttk.LabelFrame(main, text="Параметры сети", padding=8)
        settings.pack(fill="x", pady=6)

        self.activation_var = tk.StringVar(value="tanh")
        self.hidden_var = tk.StringVar(value="128,64")
        self.lr_var = tk.DoubleVar(value=0.05)
        self.epochs_var = tk.IntVar(value=20)
        self.batch_size_var = tk.IntVar(value=128)

        ttk.Label(settings, text="Активация:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(settings, textvariable=self.activation_var, values=list(ACTIVATIONS.keys()), state="readonly", width=10).grid(row=0, column=1, padx=6)
        ttk.Label(settings, text="Скрытые слои (через запятую):").grid(row=0, column=2, sticky="w")
        ttk.Entry(settings, textvariable=self.hidden_var, width=18).grid(row=0, column=3, padx=6)

        ttk.Label(settings, text="Норма обучения:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Combobox(settings, textvariable=self.lr_var, values=[0.01, 0.05, 0.1], state="readonly", width=10).grid(row=1, column=1, padx=6, pady=(8, 0))
        ttk.Label(settings, text="Эпохи:").grid(row=1, column=2, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.epochs_var, width=10).grid(row=1, column=3, padx=6, pady=(8, 0), sticky="w")
        ttk.Label(settings, text="Batch size:").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(settings, textvariable=self.batch_size_var, width=10).grid(row=1, column=5, padx=6, pady=(8, 0), sticky="w")

        actions = ttk.LabelFrame(main, text="Данные", padding=8)
        actions.pack(fill="x", pady=6)
        ttk.Button(actions, text="1) Выбрать папку эталонов", command=self.select_train_folder).grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(actions, text="2) Обучить на эталонах", command=self.train_network).grid(row=0, column=1, padx=4, pady=4, sticky="w")
        ttk.Button(actions, text="3) Загрузить изображение для распознавания", command=self.recognize_manual_image).grid(row=0, column=2, padx=4, pady=4, sticky="w")

        ttk.Button(actions, text="Выбрать папку MNIST", command=self.select_mnist_folder).grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(actions, text="Обучить на MNIST", command=self.train_on_mnist).grid(row=1, column=1, padx=4, pady=4, sticky="w")

        ttk.Button(actions, text="Папка случайных изображений", command=self.select_random_folder).grid(row=2, column=0, padx=4, pady=4, sticky="w")
        ttk.Button(actions, text="Распознать случайное из папки", command=self.recognize_random_from_folder).grid(row=2, column=1, padx=4, pady=4, sticky="w")

        self.info_label = ttk.Label(actions, text="Папка эталонов не выбрана")
        self.info_label.grid(row=3, column=0, columnspan=3, sticky="w", pady=(6, 0))

        output = ttk.PanedWindow(main, orient=tk.HORIZONTAL)
        output.pack(fill="both", expand=True)

        left = ttk.LabelFrame(output, text="Журнал")
        right = ttk.LabelFrame(output, text="Рисование и предпросмотр")
        output.add(left, weight=3)
        output.add(right, weight=2)

        self.log = tk.Text(left, wrap="word", state="disabled")
        self.log.pack(fill="both", expand=True)

        right_container = ttk.Frame(right, padding=8)
        right_container.pack(fill="both", expand=True)

        self.preview = ttk.Label(right_container, text="Здесь появится изображение из файла")
        self.preview.pack(fill="x", pady=(0, 8))

        ttk.Label(right_container, text="Нарисуйте арабскую цифру (0-9):").pack(anchor="w")
        self.draw_canvas = tk.Canvas(right_container, width=self.canvas_size, height=self.canvas_size, bg="black", highlightthickness=1, highlightbackground="#888")
        self.draw_canvas.pack(pady=6)
        self.draw_canvas.bind("<B1-Motion>", self.on_draw)
        self.draw_canvas.bind("<Button-1>", self.on_draw)

        draw_buttons = ttk.Frame(right_container)
        draw_buttons.pack(fill="x", pady=4)
        ttk.Button(draw_buttons, text="Очистить", command=self.clear_drawing).pack(side="left", padx=(0, 6))
        ttk.Button(draw_buttons, text="Распознать рисунок", command=self.recognize_drawing).pack(side="left")

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _parse_hidden_layers(self) -> List[int]:
        raw = self.hidden_var.get().strip()
        if not raw:
            return []
        return [int(x.strip()) for x in raw.split(",") if x.strip()]

    def _create_model(self, input_size: int, n_classes: int) -> None:
        hidden = self._parse_hidden_layers()
        architecture = [input_size, *hidden, n_classes]
        self.model = MLP(
            architecture,
            activation_name=self.activation_var.get(),
            learning_rate=float(self.lr_var.get()),
            seed=7,
        )
        self._append_log(f"Архитектура: {architecture}")

    def select_train_folder(self) -> None:
        selected = filedialog.askdirectory(title="Выберите папку с эталонными изображениями цифр")
        if not selected:
            return
        self.train_dir = Path(selected)
        self.info_label.config(text=f"Папка эталонов: {self.train_dir}")
        self._append_log(f"Выбрана папка эталонов: {self.train_dir}")

    def select_mnist_folder(self) -> None:
        selected = filedialog.askdirectory(title="Выберите папку с файлами MNIST (idx/idx.gz)")
        if not selected:
            return
        self.mnist_dir = Path(selected)
        self._append_log(f"Выбрана папка MNIST: {self.mnist_dir}")

    def select_random_folder(self) -> None:
        selected = filedialog.askdirectory(title="Выберите папку со случайными изображениями")
        if not selected:
            return
        self.random_dir = Path(selected)
        self._append_log(f"Выбрана папка случайных изображений: {self.random_dir}")

    def train_network(self) -> None:
        if not self.train_dir:
            messagebox.showwarning("Внимание", "Сначала выберите папку эталонов")
            return

        try:
            image_size = int(self.image_size.get())
            x, y, self.labels, files = load_dataset_from_folder(self.train_dir, image_size)
            self._create_model(x.shape[1], len(self.labels))
            self.model.fit(x, one_hot(y, len(self.labels)), epochs=int(self.epochs_var.get()), batch_size=int(self.batch_size_var.get()))
            pred = self.model.predict(x)
            train_acc = float(np.mean(pred == y))
            self._append_log(f"Обучение на эталонах завершено. Классов: {len(self.labels)}, изображений: {len(files)}")
            self._append_log(f"Точность на обучении: {train_acc:.4f}")
            messagebox.showinfo("Успех", "Сеть обучена на папке эталонов")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def train_on_mnist(self) -> None:
        if not self.mnist_dir:
            messagebox.showwarning("Внимание", "Сначала выберите папку MNIST")
            return
        try:
            x_train, y_train, x_test, y_test = load_mnist_dataset(self.mnist_dir)
            self.labels = [str(i) for i in range(10)]
            self._create_model(x_train.shape[1], 10)
            self.model.fit(
                x_train,
                one_hot(y_train, 10),
                epochs=int(self.epochs_var.get()),
                batch_size=int(self.batch_size_var.get()),
            )
            train_acc = float(np.mean(self.model.predict(x_train) == y_train))
            test_acc = float(np.mean(self.model.predict(x_test) == y_test))
            self._append_log(f"MNIST загружен: train={len(x_train)}, test={len(x_test)}")
            self._append_log(f"Точность на train: {train_acc:.4f}")
            self._append_log(f"Точность на test: {test_acc:.4f}")
            messagebox.showinfo("Успех", f"Сеть обучена на MNIST. Test accuracy: {test_acc:.4f}")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def _show_preview(self, path: Path) -> None:
        img = Image.open(path).convert("RGB")
        img.thumbnail((340, 240))
        tk_img = ImageTk.PhotoImage(img)
        self.preview.configure(image=tk_img, text="")
        self.preview.image = tk_img

    def _recognize_vector(self, vector: np.ndarray, source_name: str) -> None:
        if self.model is None:
            messagebox.showwarning("Внимание", "Сначала обучите сеть")
            return

        logits = self.model.predict_logits(vector)[0]
        idx = int(np.argmax(logits))
        confidence = float(logits[idx])
        label = self.labels[idx] if idx < len(self.labels) else str(idx)
        self._append_log(f"Распознавание: {source_name} -> {label} (оценка: {confidence:.4f})")

    def _recognize_path(self, path: Path) -> None:
        vector = image_to_vector(path, int(self.image_size.get()))
        self._recognize_vector(vector, path.name)
        self._show_preview(path)

    def recognize_manual_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите изображение для распознавания",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")],
        )
        if not path:
            return
        try:
            self._recognize_path(Path(path))
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def recognize_random_from_folder(self) -> None:
        if not self.random_dir:
            messagebox.showwarning("Внимание", "Сначала выберите папку случайных изображений")
            return
        files = [p for p in self.random_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        if not files:
            messagebox.showwarning("Внимание", "В папке нет изображений")
            return
        random_path = random.choice(files)
        try:
            self._recognize_path(random_path)
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))

    def on_draw(self, event: tk.Event) -> None:
        x, y = event.x, event.y
        r = self.brush_size // 2
        self.draw_canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")
        self.draw_ctx.ellipse((x - r, y - r, x + r, y + r), fill=255)

    def clear_drawing(self) -> None:
        self.draw_canvas.delete("all")
        self.draw_image = Image.new("L", (self.canvas_size, self.canvas_size), color=0)
        self.draw_ctx = ImageDraw.Draw(self.draw_image)
        self._append_log("Поле рисования очищено")

    def recognize_drawing(self) -> None:
        try:
            vector = pil_to_vector(self.draw_image, int(self.image_size.get()))
            self._recognize_vector(vector, "рисунок с холста")
        except Exception as exc:
            messagebox.showerror("Ошибка", str(exc))


def main() -> None:
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
