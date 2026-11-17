import random

import numpy as np

random.seed(0)
np.random.seed(0)

O_VAL, X_VAL, EMPTY_VAL = 0.01, 0.99, 0.5

WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6)
]


def is_win(cells: frozenset) -> bool:
    return any(all(c in cells for c in line) for line in WIN_LINES)


def board_to_vector(x_cells, o_cells):
    v = np.full(9, EMPTY_VAL)
    for c in x_cells:
        v[c] = X_VAL
    for c in o_cells:
        v[c] = O_VAL
    return v


_minimax_cache = {}


def minimax(x_cells: frozenset, o_cells: frozenset, turn: str):
    key = (x_cells, o_cells, turn)
    if key in _minimax_cache:
        return _minimax_cache[key]

    if is_win(x_cells):
        return 1, None
    if is_win(o_cells):
        return -1, None

    occupied = x_cells | o_cells
    if len(occupied) == 9:
        return 0, None

    empty = [c for c in range(9) if c not in occupied]
    best_move = None

    if turn == 'X':
        best_score = -2
        for c in empty:
            score, _ = minimax(x_cells | {c}, o_cells, 'O')
            if score > best_score:
                best_score, best_move = score, c
    else:
        best_score = 2
        for c in empty:
            score, _ = minimax(x_cells, o_cells | {c}, 'X')
            if score < best_score:
                best_score, best_move = score, c

    _minimax_cache[key] = (best_score, best_move)
    return best_score, best_move


def collect_o_turn_states():
    states = []
    visited = set()

    def explore(x_cells, o_cells, turn):
        key = (x_cells, o_cells, turn)
        if key in visited:
            return
        visited.add(key)

        if is_win(x_cells) or is_win(o_cells) or len(x_cells) + len(o_cells) == 9:
            return

        if turn == 'O':
            states.append((x_cells, o_cells))

        empty = [c for c in range(9) if c not in x_cells and c not in o_cells]
        for c in empty:
            if turn == 'X':
                explore(x_cells | {c}, o_cells, 'O')
            else:
                explore(x_cells, o_cells | {c}, 'X')

    explore(frozenset(), frozenset(), 'X')
    return states


def build_training_set():
    inputs, targets = [], []
    for x_cells, o_cells in collect_o_turn_states():
        _, move = minimax(x_cells, o_cells, 'O')
        if move is None:
            continue
        inp = board_to_vector(x_cells, o_cells)
        out = inp.copy()
        out[move] = O_VAL
        inputs.append(inp)
        targets.append(out)
    return np.array(inputs), np.array(targets)


class BackpropNet:
    def __init__(self, sizes, lr=0.25, momentum=0.5):
        self.sizes = sizes
        self.lr = lr
        self.momentum = momentum
        self.weights, self.biases = [], []
        self.dW_prev, self.dB_prev = [], []
        for i in range(len(sizes) - 1):
            w = np.random.uniform(-1, 1, (sizes[i], sizes[i + 1]))
            b = np.random.uniform(-1, 1, (sizes[i + 1],))
            self.weights.append(w)
            self.biases.append(b)
            self.dW_prev.append(np.zeros_like(w))
            self.dB_prev.append(np.zeros_like(b))

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(y):
        # y - уже вычисленное значение sigmoid(x)
        return y * (1.0 - y)

    def forward(self, x):
        activations = [x]
        a = x
        for w, b in zip(self.weights, self.biases):
            a = self.sigmoid(a @ w + b)
            activations.append(a)
        return activations

    def train_step(self, x, target):
        activations = self.forward(x)
        output = activations[-1]

        # ошибка и локальный градиент выходного слоя
        error = target - output
        delta = error * self.sigmoid_derivative(output)
        deltas = [delta]

        # обратное распространение ошибки по скрытым слоям
        for l in range(len(self.weights) - 1, 0, -1):
            delta = (deltas[0] @ self.weights[l].T) * self.sigmoid_derivative(activations[l])
            deltas.insert(0, delta)

        # коррекция весов и порогов (смещений)
        for l in range(len(self.weights)):
            a_prev = activations[l]
            dW = np.outer(a_prev, deltas[l])
            dB = deltas[l]
            self.weights[l] += self.lr * dW + self.momentum * self.dW_prev[l]
            self.biases[l] += self.lr * dB + self.momentum * self.dB_prev[l]
            self.dW_prev[l] = dW
            self.dB_prev[l] = dB

        return np.mean(error ** 2)

    def fit(self, X, Y, epochs=1500, verbose_every=150):
        n = len(X)
        for epoch in range(1, epochs + 1):
            order = np.random.permutation(n)
            total_err = 0.0
            for i in order:
                total_err += self.train_step(X[i], Y[i])
            if verbose_every and epoch % verbose_every == 0:
                print(f"Эпоха {epoch:5d}   средняя квадратичная ошибка = {total_err / n:.6f}")

    def predict(self, x):
        return self.forward(x)[-1]


def net_move(net: BackpropNet, board_vec: np.ndarray):
    empty_cells = [i for i in range(9) if abs(board_vec[i] - EMPTY_VAL) < 1e-6]
    if not empty_cells:
        return None
    out = net.predict(board_vec)
    return min(empty_cells, key=lambda i: out[i])


def print_board(x_cells, o_cells):
    symbols = []
    for i in range(9):
        if i in x_cells:
            symbols.append('X')
        elif i in o_cells:
            symbols.append('O')
        else:
            symbols.append('.')
    for r in range(3):
        print(' '.join(symbols[r * 3:(r + 1) * 3]))
    print()


def play_demo_game(net: BackpropNet, x_strategy='random'):
    x_cells, o_cells = set(), set()
    turn = 'X'
    print("Начальная позиция:")
    print_board(x_cells, o_cells)

    while True:
        empty = [c for c in range(9) if c not in x_cells and c not in o_cells]

        if turn == 'X':
            if x_strategy == 'random':
                move = random.choice(empty)
            else:
                _, move = minimax(frozenset(x_cells), frozenset(o_cells), 'X')
            x_cells.add(move)
            print(f"Крестик ходит в клетку {move + 1}")
        else:
            vec = board_to_vector(frozenset(x_cells), frozenset(o_cells))
            move = net_move(net, vec)
            o_cells.add(move)
            print(f"Нолик (сеть) ходит в клетку {move + 1}")

        print_board(x_cells, o_cells)

        if is_win(frozenset(x_cells)):
            print("Победили крестики!")
            return
        if is_win(frozenset(o_cells)):
            print("Победили нолики (сеть)!")
            return
        if len(x_cells) + len(o_cells) == 9:
            print("Ничья.")
            return

        turn = 'O' if turn == 'X' else 'X'


if __name__ == "__main__":
    print("Формирование обучающей выборки")
    X_train, Y_train = build_training_set()
    print(f"Размер обучающей выборки: {len(X_train)} примеров\n")

    net = BackpropNet(sizes=[9, 27, 9], lr=0.25, momentum=0.5)

    print("Обучение сети методом обратного распространения ошибки...")
    net.fit(X_train, Y_train, epochs=1500, verbose_every=150)

    print("\nПроверка обученной сети на нескольких показательных партиях:")
    for game_num in range(1, 4):
        print(f"\n=== Партия {game_num} (крестик ходит случайно) ===")
        random.seed(100 + game_num)
        play_demo_game(net, x_strategy='random')
