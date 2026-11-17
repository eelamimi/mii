import numpy as np

kurs_euro = []
with open("euro.csv", "r") as f:
    f.readline()
    for r in f.readlines():
        splitted = r.split(';')
        kurs_euro.append((splitted[1], float(splitted[2].replace(',', '.'))))
kurs_euro = kurs_euro[::-1]

prices = np.array([k[1] for k in kurs_euro])
diffs = np.diff(prices)


def act(s):
    return 2.0 / (1.0 + np.exp(-s)) - 1.0


def act_deriv(a):
    return 0.5 * (1.0 + a) * (1.0 - a)


N_IN = 10
N_HID = 3
LR = 0.05
EPOCHS = 500

X = np.array([diffs[i:i + N_IN] for i in range(len(diffs) - N_IN)])
y = np.array([diffs[i + N_IN] for i in range(len(diffs) - N_IN)])

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

max_abs = np.max(np.abs(diffs[:split + N_IN]))


def to_net(v):
    return 0.9 * v / max_abs


def from_net(v):
    return v / 0.9 * max_abs


Xn_train, Xn_test = to_net(X_train), to_net(X_test)
yn_train = to_net(y_train).reshape(-1, 1)

rng = np.random.default_rng(1)
W1 = rng.uniform(-0.5, 0.5, (N_IN, N_HID))
b1 = np.zeros(N_HID)
W2 = rng.uniform(-0.5, 0.5, (N_HID, 1))
b2 = np.zeros(1)


def forward(x):
    h = act(x @ W1 + b1)
    o = act(h @ W2 + b2)
    return h, o


best_val = np.inf
best = None
Xn_test_t = Xn_test
yn_test = to_net(y_test).reshape(-1, 1)

for epoch in range(EPOCHS):
    idx = rng.permutation(len(Xn_train))
    for i in idx:
        x = Xn_train[i:i + 1]
        t = yn_train[i:i + 1]
        h, o = forward(x)
        err = o - t
        d_out = err * act_deriv(o)
        d_hid = (d_out @ W2.T) * act_deriv(h)
        W2 -= LR * h.T @ d_out
        b2 -= LR * d_out[0]
        W1 -= LR * x.T @ d_hid
        b1 -= LR * d_hid[0]

    val = np.mean((forward(Xn_test_t)[1] - yn_test) ** 2)
    if val < best_val:
        best_val = val
        best = (W1.copy(), b1.copy(), W2.copy(), b2.copy())

    if (epoch + 1) % 250 == 0:
        print(f"Эпоха {epoch + 1:5d}, val MSE = {val:.6f}, лучший = {best_val:.6f}")

W1, b1, W2, b2 = best


def predict(x_raw):
    return from_net(forward(to_net(x_raw))[1]).ravel()


def report(name, pred, true):
    mae = np.mean(np.abs(pred - true))
    mae_zero = np.mean(np.abs(true))
    direction = np.mean(np.sign(pred) == np.sign(true)) * 100
    print(f"{name}: MAE сети = {mae:.4f} руб., MAE «нулевого» прогноза = "
          f"{mae_zero:.4f} руб., верное направление = {direction:.1f}%")


report("Обучение", predict(X_train), y_train)
report("Проверка", predict(X_test), y_test)

last10 = diffs[-N_IN:]
d_next = predict(last10.reshape(1, -1))[0]
print(f"\nПоследний курс: {prices[-1]:.4f}")
print(f"Прогноз изменения: {d_next:+.4f} руб.")
print(f"Прогнозируемый курс на следующий день: {prices[-1] + d_next:.4f}")
