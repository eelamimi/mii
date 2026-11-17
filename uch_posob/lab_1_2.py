import numpy as np

np.random.seed(42)

X_BASE = np.array([
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [1, 0, 0, 0, 1],
], dtype=int)

O_BASE = np.array([
    [0, 1, 1, 1, 0],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1],
    [0, 1, 1, 1, 0],
], dtype=int)


def show(image_5x5, symbol="#"):
    for row in image_5x5:
        print(" ".join(symbol if v else "." for v in row))
    print()


def add_noise(image, n_flip, rng):
    img = image.copy().flatten()
    idx = rng.choice(len(img), size=n_flip, replace=False)
    img[idx] = 1 - img[idx]
    return img.reshape(image.shape)


rng = np.random.default_rng(1)

train_images = []
train_labels = []

train_images.append(X_BASE)
train_labels.append(1)
train_images.append(O_BASE)
train_labels.append(0)

for _ in range(6):
    n_flip = rng.integers(1, 3)
    train_images.append(add_noise(X_BASE, n_flip, rng))
    train_labels.append(1)

for _ in range(6):
    n_flip = rng.integers(1, 3)
    train_images.append(add_noise(O_BASE, n_flip, rng))
    train_labels.append(0)

X_train = np.array([img.flatten() for img in train_images], dtype=float)
y_train = np.array(train_labels, dtype=float)

print(f"Сформирована обучающая выборка: {X_train.shape[0]} образов "
      f"(по {int(y_train.sum())} крестиков и {int((1 - y_train).sum())} ноликов), "
      f"каждый вектор длиной {X_train.shape[1]}.\n")


class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.2, seed=0):
        rng_local = np.random.default_rng(seed)
        # небольшие случайные начальные веса и смещение (bias)
        self.weights = rng_local.uniform(-0.1, 0.1, size=n_inputs)
        self.bias = 0.0
        self.lr = learning_rate

    def net_input(self, x):
        return np.dot(self.weights, x) + self.bias

    def predict(self, x):
        return 1.0 if self.net_input(x) >= 0 else 0.0

    def train(self, X, y, max_epochs=100, verbose=True):
        for epoch in range(1, max_epochs + 1):
            errors = 0
            for xi, target in zip(X, y):
                output = self.predict(xi)
                error = target - output
                if error != 0:
                    self.weights += self.lr * error * xi
                    self.bias += self.lr * error
                    errors += 1
            if verbose:
                print(f"Эпоха {epoch:3d}: ошибок = {errors}")
            if errors == 0:
                if verbose:
                    print(f"\nОбучение завершено на эпохе {epoch}: "
                          f"персептрон безошибочно классифицирует все образы.\n")
                return epoch
        print("\nДостигнуто максимальное число эпох, полной сходимости не получено.\n")
        return max_epochs


perceptron = Perceptron(n_inputs=25, learning_rate=0.2, seed=0)
epochs_used = perceptron.train(X_train, y_train, max_epochs=50, verbose=True)

print("=" * 50)
print("Проверка на обучающей выборке:")
print("=" * 50)
correct = 0
for i, (xi, target) in enumerate(zip(X_train, y_train), start=1):
    out = perceptron.predict(xi)
    ok = "OK" if out == target else "ОШИБКА"
    correct += (out == target)
    label_name = "крестик" if target == 1 else "нолик "
    print(f"Образ {i:2d} ({label_name}): выход сети = {int(out)}, "
          f"ожидалось = {int(target)}  -> {ok}")
print(f"\nВерно распознано {correct} из {len(y_train)} обучающих образов.\n")

print("=" * 50)
print("Проверка на новых тестовых изображениях (не участвовавших в обучении):")
print("=" * 50)

rng_test = np.random.default_rng(99)
test_cases = []
for _ in range(2):
    test_cases.append((add_noise(X_BASE, 2, rng_test), 1, "крестик"))
for _ in range(2):
    test_cases.append((add_noise(O_BASE, 2, rng_test), 0, "нолик"))

for i, (img, target, name) in enumerate(test_cases, start=1):
    print(f"--- Тестовый образ {i} ({name}) ---")
    show(img)
    out = perceptron.predict(img.flatten().astype(float))
    ok = "OK" if out == target else "ОШИБКА"
    print(f"Выход персептрона = {int(out)} (ожидалось {target}) -> {ok}\n")

print("Итоговые веса персептрона (форма 5x5):")
print(np.round(perceptron.weights, 2).reshape(5, 5))
print(f"Смещение (bias): {perceptron.bias:.3f}")
