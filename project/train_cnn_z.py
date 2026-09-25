import numpy as np
import pandas as pd
from pathlib import Path
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

CSV_PATH = Path("df_training.csv")
IMAGE_SIZE = 8
RANDOM_SEED = 1234
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
NORMALIZE = "max"  # "sum", "max", "log1p", or None
TARGET_COLUMN = "z_gamma"
Z_MIN = -7.5
Z_MAX = 7.5
OUT_OF_BOUNDS_PENALTY = 0.0

def set_reproducible(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(1)

def prep_data():
    # -------- Load data --------
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Identify PE columns
    pe_cols = [c for c in df.columns if c.startswith("pe_")]
    print(pe_cols)
    if len(pe_cols) == 0:
        raise RuntimeError("No pe_ columns found in CSV")

    # Build 8x8 images
    # Expect columns like pe_0_0 ... pe_7_7
    pe_img = np.zeros((len(df), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for col in pe_cols:
        try:
            _, ix, iy = col.split("_")
            ix = int(ix)
            iy = int(iy)
        except Exception:
            continue
        if 0 <= ix < IMAGE_SIZE and 0 <= iy < IMAGE_SIZE:
            pe_img[:, IMAGE_SIZE - 1 - iy, ix] = df[col].to_numpy(dtype=np.float32) # have to covert from physical indexing of input to np indexing

    # Optional normalization per event
    if NORMALIZE == "log1p":
        pe_img = np.log1p(pe_img)
    elif NORMALIZE == "sum":
        denom = pe_img.sum(axis=(1, 2), keepdims=True)
        denom[denom == 0] = 1.0
        pe_img = pe_img / denom
    elif NORMALIZE == "max":
        denom = pe_img.max(axis=(1, 2), keepdims=True)
        denom[denom == 0] = 1.0
        pe_img = pe_img / denom
    elif NORMALIZE is not None:
        raise ValueError(f"Unknown NORMALIZE mode: {NORMALIZE}")

    print("pe img:")
    print(pe_img[0,:,:])

    # Target
    if TARGET_COLUMN not in df.columns:
        raise RuntimeError(f"Missing target column: {TARGET_COLUMN}")

    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    # Train/val/test split
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(df))
    rng.shuffle(idx)

    n = len(idx)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    idx_train = idx[:train_end]
    idx_val = idx[train_end:val_end]
    idx_test = idx[val_end:]

    X_train = pe_img[idx_train]
    X_val = pe_img[idx_val]
    X_test = pe_img[idx_test]

    print(X_train.shape)
    y_train = y[idx_train]
    y_val = y[idx_val]
    y_test = y[idx_test]

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_cnn(X_train, X_val, X_test, y_train, y_val, y_test):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_nchw(x):
        if x.ndim == 3:
            return x[:, None, :, :]
        if x.ndim == 4:
            return x
        raise ValueError(f"Unexpected input shape {x.shape}; expected (N,H,W) or (N,C,H,W)")

    # Convert to NCHW for training
    X_train = _ensure_nchw(X_train)
    X_val = _ensure_nchw(X_val)
    X_test = _ensure_nchw(X_test)

    in_ch = X_train.shape[1]

    class SimpleCNN(nn.Module):
        def __init__(self, in_channels: int):
            super().__init__()
            self.branch_k2 = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=2, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(2),
            )
            self.branch_k4 = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=4, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(2),
            )
            self.branch_k8 = nn.Sequential(
                nn.Conv2d(in_channels, 8, kernel_size=8, padding=0),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(2),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(8 * 3 * 2 * 2, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )

        def forward(self, x):
            b2 = self.branch_k2(x)
            b4 = self.branch_k4(x)
            b8 = self.branch_k8(x)
            feats = torch.cat([b2, b4, b8], dim=1)
            return self.head(feats).squeeze(1)

    model = SimpleCNN(in_ch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # DataLoaders
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # -------- Train --------
    best_val = float("inf")
    history = {"epoch": [], "train_mse": [], "val_mse": []}
    for epoch in range(1, EPOCHS + 1):
            model.train()
            train_loss = 0.0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                mse = loss_fn(pred, yb)
                penalty = (torch.relu(pred - Z_MAX)**2 + torch.relu(Z_MIN - pred)**2).mean()
                loss = mse + OUT_OF_BOUNDS_PENALTY * penalty
                loss.backward()
                opt.step()
                train_loss += loss.item() * xb.size(0)

            train_loss /= len(train_ds)

            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_ds)

            print(f"Epoch {epoch:02d} | train MSE={train_loss:.4f} | val MSE={val_loss:.4f}")
            history["epoch"].append(epoch)
            history["train_mse"].append(train_loss)
            history["val_mse"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), "cnn_z_best.pt")
                #print("Saved best model to cnn_z_best.pt")

    # Training curves
    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_mse"], label="Train MSE")
    plt.plot(history["epoch"], history["val_mse"], label="Val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig("cnn-performance/training-curves.png")

    # -------- Evaluate --------
    model.load_state_dict(torch.load("cnn_z_best.pt", map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())

    preds = np.concatenate(preds)

    return y_test, preds

def evaluate(y_test, preds):

    mae = np.mean(np.abs(preds - y_test))
    rmse = np.sqrt(np.mean((preds - y_test) ** 2))
    print(f"Test MAE: {mae:.4f} mm | RMSE: {rmse:.4f} mm")

    # Scatter: Z_pred vs Z_true (test set)
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, preds, s=6, alpha=0.4)
    minv = float(min(y_test.min(), preds.min()))
    maxv = float(max(y_test.max(), preds.max()))
    #plt.plot([minv, maxv], [minv, maxv], 'r--', linewidth=1)
    plt.xlabel("Z_true (mm)")
    plt.ylabel("Z_pred (mm)")
    plt.title("Z_pred vs Z_true (test set)")
    plt.tight_layout()
    plt.savefig("cnn-performance/scatter.png")

    test_resid = y_test - preds

    plt.figure(figsize=(6, 4))
    counts, edges, _ = plt.hist(test_resid, bins=40, alpha=0.85)
    plt.xlabel("Z_true - Z_pred (mm)")
    plt.ylabel("Counts")
    plt.title("residuals (test set)")

    # Gaussian fit (counts vs bin centers)
    centers = 0.5 * (edges[:-1] + edges[1:])
    def _gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    A0 = counts.max() if len(counts) > 0 else 1.0
    mu0 = float(np.mean(test_resid)) if len(test_resid) > 0 else 0.0
    sigma0 = float(np.std(test_resid)) if len(test_resid) > 0 else 1.0
    resid_std = float(np.std(test_resid)) if len(test_resid) > 0 else np.nan

    fit_ok = False
    mu_err = np.nan

    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(_gauss, centers, counts, p0=[A0, mu0, sigma0], maxfev=20000)
        A, mu, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        mu_err = perr[1] if len(perr) > 1 else np.nan
        fit_ok = True
    except Exception:
        A, mu, sigma = A0, mu0, sigma0

    fwhm = 2.355 * sigma if sigma is not None else np.nan

    xfit = np.linspace(edges[0], edges[-1], 200) if len(edges) > 1 else np.array([0.0, 1.0])
    yfit = _gauss(xfit, A, mu, sigma) if sigma is not None and sigma != 0 else np.zeros_like(xfit)
    plt.plot(xfit, yfit, 'r-', label='Gaussian fit' if fit_ok else 'Gaussian (moment est.)')

    txt = (
        f"Data std = {resid_std:.3g}\n"
        f"$\\mu$ = {mu:.3g} ± {mu_err:.2g}\n"
        f"$\\sigma$ = {sigma:.3g}\n"
        f"FWHM = {fwhm:.3g}"
    )
    plt.text(0.98, 0.95, txt, transform=plt.gca().transAxes, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
    plt.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig("cnn-performance/pull.png")


set_reproducible(RANDOM_SEED)
X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
y_test, preds = train_cnn(X_train, X_val, X_test, y_train, y_val, y_test)
evaluate(y_test, preds)
