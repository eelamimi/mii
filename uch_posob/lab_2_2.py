import argparse
import glob
import os
import random
import string
from collections import Counter

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import DataLoader, Dataset

IMG = 20
N_IN = IMG * IMG
LETTERS = string.ascii_uppercase
N_CLASSES = len(LETTERS)
N_BITS = 5

BAD_NAMES = ("symbol", "wingding", "webding", "dingbat", "marlett", "ornament",
             "emoji", "icon", "bats", "music", "dings", "braille", "cursor", "mt extra")


def glyph_image(font, ch, augment=False):
    canvas = Image.new("L", (160, 160), 0)
    ImageDraw.Draw(canvas).text((40, 30), ch, fill=255, font=font)

    if augment:
        shear = random.uniform(-0.35, 0.35)
        canvas = canvas.transform(canvas.size, Image.Transform.AFFINE,
                                  (1, shear, -shear * 80, 0, 1, 0),
                                  Image.Resampling.BILINEAR)
        canvas = canvas.rotate(random.uniform(-8, 8), resample=Image.Resampling.BILINEAR)
        mode = random.choice(["none", "none", "thick", "thick2", "thin"])
        if mode == "thick":
            canvas = canvas.filter(ImageFilter.MaxFilter(3))
        elif mode == "thick2":
            canvas = canvas.filter(ImageFilter.MaxFilter(5))
        elif mode == "thin":
            canvas = canvas.filter(ImageFilter.MinFilter(3))

    bbox = canvas.getbbox()
    if bbox is None:
        return None
    glyph = canvas.crop(bbox)
    w, h = glyph.size
    if augment:
        w = max(1, int(w * random.uniform(0.8, 1.2)))
        h = max(1, int(h * random.uniform(0.8, 1.2)))
        glyph = glyph.resize((w, h), Image.Resampling.BILINEAR)
    side = max(w, h)
    square = Image.new("L", (side, side), 0)
    square.paste(glyph, ((side - w) // 2, (side - h) // 2))
    square = square.resize((IMG, IMG), Image.Resampling.LANCZOS)
    arr = np.asarray(square, dtype=np.float32) / 255.0
    return np.clip(arr, 0.0, 1.0)


def font_is_good(path):
    try:
        font = ImageFont.truetype(path, 64)
        name = " ".join(font.getname()).lower() + " " + os.path.basename(path).lower()
        if any(b in name for b in BAD_NAMES):
            return None
        imgs = [glyph_image(font, ch) for ch in LETTERS]
        if any(i is None for i in imgs):
            return None
        if len({i.tobytes() for i in imgs}) < 25:
            return None
        return font.getname()
    except Exception:
        return None


def find_fonts(extra_dir=None, limit=80):
    dirs = [extra_dir] if extra_dir else []
    dirs += ["C:/Windows/Fonts", "/usr/share/fonts", "/usr/local/share/fonts",
             os.path.expanduser("~/.fonts"), "/Library/Fonts", "/System/Library/Fonts",
             os.path.join(matplotlib.get_data_path(), "fonts", "ttf")]
    found = []
    for d in dirs:
        if d and os.path.isdir(d):
            found += glob.glob(os.path.join(d, "**", "*.ttf"), recursive=True)
    found = sorted(set(found))
    random.Random(0).shuffle(found)

    good, seen = [], set()
    for p in found:
        name = font_is_good(p)
        if name and name not in seen:
            seen.add(name)
            good.append(p)
        if len(good) >= limit:
            break
    return good


def shift(arr, dx, dy):
    out = np.zeros_like(arr)
    ys, yd = (slice(dy, None), slice(0, IMG - dy)) if dy >= 0 else (slice(0, IMG + dy), slice(-dy, None))
    xs, xd = (slice(dx, None), slice(0, IMG - dx)) if dx >= 0 else (slice(0, IMG + dx), slice(-dx, None))
    out[ys, xs] = arr[yd, xd]
    return out


def build_tensors(font_paths, n_aug):
    X, y = [], []
    for p in font_paths:
        font = ImageFont.truetype(p, 64)
        for idx, ch in enumerate(LETTERS):
            X.append(glyph_image(font, ch).reshape(-1))
            y.append(idx)
            for _ in range(n_aug):
                img = glyph_image(font, ch, augment=True)
                if img is None:
                    continue
                img = shift(img, random.randint(-1, 1), random.randint(-1, 1))
                X.append(img.reshape(-1))
                y.append(idx)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(y, dtype=torch.long))


class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class Perceptron(nn.Module):
    def __init__(self, in_features, hidden, out_features, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, out_features),
        )

    def forward(self, x):
        return self.net(x)


def make_loss(mode, device):
    shifts = torch.arange(N_BITS - 1, -1, -1, device=device)
    if mode == "onehot":
        ce = nn.CrossEntropyLoss(label_smoothing=0.05)
        return lambda logits, y: ce(logits, y)
    bce = nn.BCEWithLogitsLoss()
    return lambda logits, y: bce(logits, ((y.unsqueeze(1) >> shifts) & 1).float())


def decode(logits, mode):
    if mode == "onehot":
        return logits.argmax(dim=1)
    bits = (torch.sigmoid(logits) > 0.5).long()
    weights = 2 ** torch.arange(N_BITS - 1, -1, -1, device=logits.device)
    return (bits * weights).sum(dim=1)


def evaluate(model, loader, loss_fn, mode, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += loss_fn(logits, yb).item() * xb.size(0)
            correct += (decode(logits, mode) == yb).sum().item()
            n += xb.size(0)
    return total / n, correct / n


def show_glyph(vec):
    for row in vec.reshape(IMG, IMG):
        print("".join("█" if v > 0.5 else ("▒" if v > 0.2 else "·") for v in row))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fonts-dir", default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--max-fonts", type=int, default=80)
    ap.add_argument("--n-aug", type=int, default=15, help="случайных вариантов на букву")
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--output", choices=["onehot", "binary"], default="onehot")
    args = ap.parse_args()

    torch.manual_seed(0)
    random.seed(0)

    fonts = find_fonts(args.fonts_dir, args.max_fonts)
    print(f"Найдено рабочих шрифтов: {len(fonts)}")

    n_val = n_test = max(3, int(len(fonts) * 0.15))
    train_fonts = fonts[:-(n_val + n_test)]
    val_fonts = fonts[-(n_val + n_test):-n_test]
    test_fonts = fonts[-n_test:]
    print(f"Шрифтов: train={len(train_fonts)}, val={len(val_fonts)}, test={len(test_fonts)}")
    print("Тестовые шрифты:", [ImageFont.truetype(p, 20).getname()[0] for p in test_fonts])

    X_tr, y_tr = build_tensors(train_fonts, args.n_aug)
    X_val, y_val = build_tensors(val_fonts, 0)
    X_te, y_te = build_tensors(test_fonts, 0)
    print(f"Примеров: train={len(X_tr)}, val={len(X_val)}, test={len(X_te)}")

    loader = DataLoader(MyDataset(X_tr, y_tr), batch_size=64, shuffle=True,
                        num_workers=0, drop_last=False, pin_memory=False)
    val_loader = DataLoader(MyDataset(X_val, y_val), batch_size=256)
    test_loader = DataLoader(MyDataset(X_te, y_te), batch_size=256)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dim = N_CLASSES if args.output == "onehot" else N_BITS
    model = Perceptron(N_IN, args.hidden, out_dim, args.dropout).to(device)

    criterion = make_loss(args.output, device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)

    best_acc, best_loss, best_state = -1.0, 1e9, None
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
        scheduler.step()

        val_loss, val_acc = evaluate(model, val_loader, criterion, args.output, device)
        if val_acc > best_acc or (val_acc == best_acc and val_loss < best_loss):
            best_acc, best_loss = val_acc, val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d}/{args.epochs} loss={running / len(loader.dataset):.4f} "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}")

    model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, criterion, args.output, device)
    print(f"\nЛучшая val-точность: {best_acc * 100:.1f}%")
    print(f"ИТОГ на тестовых (незнакомых) шрифтах: {test_acc * 100:.1f}% букв")

    model.eval()
    with torch.no_grad():
        pred = decode(model(X_te.to(device)), args.output).cpu()
    wrong = [i for i in range(len(X_te)) if pred[i] != y_te[i]]
    print(f"Ошибок: {len(wrong)} из {len(X_te)}")
    conf = Counter((LETTERS[y_te[i]], LETTERS[pred[i]] if pred[i] < N_CLASSES else "?") for i in wrong)
    print("Частые путаницы (истина -> ответ):",
          ", ".join(f"{a}->{b} x{c}" for (a, b), c in conf.most_common(8)))
    for i in wrong[:2]:
        p = LETTERS[pred[i]] if pred[i] < N_CLASSES else "?"
        print(f"\nИстина: {LETTERS[y_te[i]]}, сеть ответила: {p}")
        show_glyph(X_te[i].numpy())


if __name__ == "__main__":
    main()
