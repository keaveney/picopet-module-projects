import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

IMAGE_SIZE = 8
PIXEL_SIZE_MM = 3.0
PIXEL_GAP_MM = 0.2
PIXEL_PITCH_MM = PIXEL_SIZE_MM + PIXEL_GAP_MM
PIXEL_OFFSET_MM = 0.5 * (IMAGE_SIZE - 1) * PIXEL_PITCH_MM
XY_HALF_WIDTH_MM = PIXEL_OFFSET_MM + 0.5 * PIXEL_SIZE_MM
MIN_EVENTS_PER_PIXEL_PLOT = 5

RANDOM_SEED = 1234
BATCH_SIZE = 128
EPOCHS = 40
LR = 1e-3
NORMALIZE = "max"  # "sum", "max", "log1p", or "none"
TARGET_COLUMNS = ("x_gamma", "y_gamma", "z_gamma")
AXIS_NAMES = ("x", "y", "z")

X_MIN = -XY_HALF_WIDTH_MM
X_MAX = XY_HALF_WIDTH_MM
Y_MIN = -XY_HALF_WIDTH_MM
Y_MAX = XY_HALF_WIDTH_MM
Z_MIN = -7.5
Z_MAX = 7.5

OUT_OF_BOUNDS_RELATIVE_WEIGHT = 0.0
OUT_OF_BOUNDS_EPS = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True, help="Merged training CSV")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for model/plots")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--normalize", type=str, default=NORMALIZE, choices=["sum", "max", "log1p", "none"])
    parser.add_argument("--target-columns", nargs=3, default=TARGET_COLUMNS, help="Target columns: x y z")
    parser.add_argument("--test-mode", action="store_true", help="Use a small random subset for quick debugging")
    parser.add_argument("--test-mode-events", type=int, default=2000, help="Maximum rows used when --test-mode is set")

    parser.add_argument("--x-min", type=float, default=X_MIN)
    parser.add_argument("--x-max", type=float, default=X_MAX)
    parser.add_argument("--y-min", type=float, default=Y_MIN)
    parser.add_argument("--y-max", type=float, default=Y_MAX)
    parser.add_argument("--z-min", type=float, default=Z_MIN)
    parser.add_argument("--z-max", type=float, default=Z_MAX)

    parser.add_argument(
        "--oob-relative-weight",
        type=float,
        default=OUT_OF_BOUNDS_RELATIVE_WEIGHT,
        help="Default relative OOB penalty strength for all axes.",
    )
    parser.add_argument("--x-oob-relative-weight", type=float, default=None)
    parser.add_argument("--y-oob-relative-weight", type=float, default=None)
    parser.add_argument("--z-oob-relative-weight", type=float, default=None)
    return parser.parse_args()


def set_reproducible(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)


def prep_data(csv_path: Path, seed: int, normalize: str, target_columns, test_mode: bool, test_mode_events: int):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    pe_cols = [c for c in df.columns if c.startswith("pe_")]
    if len(pe_cols) == 0:
        raise RuntimeError("No pe_ columns found in CSV")

    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")

    pe_img = np.zeros((len(df), IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for col in pe_cols:
        try:
            _, ix, iy = col.split("_")
            ix = int(ix)
            iy = int(iy)
        except Exception:
            continue
        if 0 <= ix < IMAGE_SIZE and 0 <= iy < IMAGE_SIZE:
            pe_img[:, IMAGE_SIZE - 1 - iy, ix] = df[col].to_numpy(dtype=np.float32)

    if normalize == "log1p":
        pe_img = np.log1p(pe_img)
    elif normalize == "sum":
        denom = pe_img.sum(axis=(1, 2), keepdims=True)
        denom[denom == 0] = 1.0
        pe_img = pe_img / denom
    elif normalize == "max":
        denom = pe_img.max(axis=(1, 2), keepdims=True)
        denom[denom == 0] = 1.0
        pe_img = pe_img / denom
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalization mode: {normalize}")

    y = df[list(target_columns)].to_numpy(dtype=np.float32)

    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    if test_mode:
        n_keep = min(test_mode_events, len(idx))
        idx = idx[:n_keep]
        print(f"TEST MODE: using {n_keep} rows out of {len(df)}")

    n = len(idx)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)

    idx_train = idx[:train_end]
    idx_val = idx[train_end:val_end]
    idx_test = idx[val_end:]

    return (
        pe_img[idx_train],
        pe_img[idx_val],
        pe_img[idx_test],
        y[idx_train],
        y[idx_val],
        y[idx_test],
        df.iloc[idx_test].reset_index(drop=True),
    )


class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 3):
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
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        b2 = self.branch_k2(x)
        b4 = self.branch_k4(x)
        b8 = self.branch_k8(x)
        feats = torch.cat([b2, b4, b8], dim=1)
        return self.head(feats)


def cartesian_distance_loss(pred, target):
    return torch.linalg.vector_norm(pred - target, dim=1).mean()


def oob_penalty_terms_xyz(pred, base_loss, bounds, relative_weights):
    weighted_total = pred.new_tensor(0.0)
    raw_terms = {}
    weighted_terms = {}

    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        lo = bounds[axis_idx, 0]
        hi = bounds[axis_idx, 1]
        rel_weight = float(relative_weights[axis_idx])

        raw_penalty = (
            torch.relu(pred[:, axis_idx] - hi) ** 2
            + torch.relu(lo - pred[:, axis_idx]) ** 2
        ).mean()

        if rel_weight <= 0:
            weighted_penalty = pred.new_tensor(0.0)
        else:
            # The scale is detached deliberately: it makes the numeric loss
            # contribution interpretable relative to the Cartesian distance
            # loss, while keeping the gradient direction tied to the boundary
            # violation itself.
            scale = base_loss.detach() / raw_penalty.detach().clamp_min(OUT_OF_BOUNDS_EPS)
            weighted_penalty = rel_weight * scale * raw_penalty

        raw_terms[axis_name] = raw_penalty
        weighted_terms[axis_name] = weighted_penalty
        weighted_total = weighted_total + weighted_penalty

    return raw_terms, weighted_terms, weighted_total


def train_cnn(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test,
    outdir: Path,
    seed: int,
    batch_size: int,
    epochs: int,
    lr: float,
    bounds_np,
    oob_relative_weights_np,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    def _ensure_nchw(x):
        if x.ndim == 3:
            return x[:, None, :, :]
        if x.ndim == 4:
            return x
        raise ValueError(f"Unexpected input shape {x.shape}")

    X_train = _ensure_nchw(X_train)
    X_val = _ensure_nchw(X_val)
    X_test = _ensure_nchw(X_test)

    bounds = torch.tensor(bounds_np, dtype=torch.float32, device=device)
    oob_relative_weights = np.asarray(oob_relative_weights_np, dtype=np.float32)

    model = SimpleCNN(X_train.shape[1], out_dim=3).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    best_val = float("inf")
    best_model_path = outdir / "cnn_xyz_best.pt"
    history = {
        "epoch": [],
        "train_distance": [],
        "train_oob_weighted": [],
        "train_total": [],
        "val_distance": [],
        "val_oob_weighted": [],
        "val_total": [],
    }
    for axis_name in AXIS_NAMES:
        history[f"train_oob_{axis_name}"] = []
        history[f"val_oob_{axis_name}"] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_dist = 0.0
        train_oob = 0.0
        train_total = 0.0
        train_oob_axis = {a: 0.0 for a in AXIS_NAMES}

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            pred = model(xb)
            distance = cartesian_distance_loss(pred, yb)
            _, oob_weighted, oob_total = oob_penalty_terms_xyz(
                pred, distance, bounds, oob_relative_weights
            )
            loss = distance + oob_total
            loss.backward()
            opt.step()

            n = xb.size(0)
            train_dist += distance.item() * n
            train_oob += oob_total.item() * n
            train_total += loss.item() * n
            for axis_name in AXIS_NAMES:
                train_oob_axis[axis_name] += oob_weighted[axis_name].item() * n

        train_dist /= len(train_ds)
        train_oob /= len(train_ds)
        train_total /= len(train_ds)
        for axis_name in AXIS_NAMES:
            train_oob_axis[axis_name] /= len(train_ds)

        model.eval()
        val_dist = 0.0
        val_oob = 0.0
        val_total = 0.0
        val_oob_axis = {a: 0.0 for a in AXIS_NAMES}

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                distance = cartesian_distance_loss(pred, yb)
                _, oob_weighted, oob_total = oob_penalty_terms_xyz(
                    pred, distance, bounds, oob_relative_weights
                )
                loss = distance + oob_total

                n = xb.size(0)
                val_dist += distance.item() * n
                val_oob += oob_total.item() * n
                val_total += loss.item() * n
                for axis_name in AXIS_NAMES:
                    val_oob_axis[axis_name] += oob_weighted[axis_name].item() * n

        val_dist /= len(val_ds)
        val_oob /= len(val_ds)
        val_total /= len(val_ds)
        for axis_name in AXIS_NAMES:
            val_oob_axis[axis_name] /= len(val_ds)

        train_oob_frac = train_oob / train_dist if train_dist > 0 else 0.0
        val_oob_frac = val_oob / val_dist if val_dist > 0 else 0.0
        print(
            f"Epoch {epoch:02d} | "
            f"train D={train_dist:.4f} mm | train OOB={train_oob:.4f} "
            f"({train_oob_frac:.1%} of D) | "
            f"val D={val_dist:.4f} mm | val OOB={val_oob:.4f} "
            f"({val_oob_frac:.1%} of D)"
        )

        history["epoch"].append(epoch)
        history["train_distance"].append(train_dist)
        history["train_oob_weighted"].append(train_oob)
        history["train_total"].append(train_total)
        history["val_distance"].append(val_dist)
        history["val_oob_weighted"].append(val_oob)
        history["val_total"].append(val_total)
        for axis_name in AXIS_NAMES:
            history[f"train_oob_{axis_name}"].append(train_oob_axis[axis_name])
            history[f"val_oob_{axis_name}"].append(val_oob_axis[axis_name])

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_model_path)

    plt.figure(figsize=(6, 4))
    plt.plot(history["epoch"], history["train_distance"], label="Train mean 3D error")
    plt.plot(history["epoch"], history["val_distance"], label="Val mean 3D error")
    if np.any(oob_relative_weights > 0):
        plt.plot(history["epoch"], history["train_total"], "--", label="Train total")
        plt.plot(history["epoch"], history["val_total"], "--", label="Val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (mm)")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training-curves.png")
    plt.close()

    if np.any(oob_relative_weights > 0):
        plt.figure(figsize=(7, 4))
        for axis_name in AXIS_NAMES:
            plt.plot(history["epoch"], history[f"val_oob_{axis_name}"], label=f"Val OOB {axis_name}")
        plt.xlabel("Epoch")
        plt.ylabel("Weighted OOB loss (mm)")
        plt.title("Validation OOB penalty by axis")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / "oob-penalty-curves.png")
        plt.close()

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            preds.append(model(xb).cpu().numpy())

    preds = np.concatenate(preds)
    return y_test, preds, history, best_val


def plot_scatter_1d(y_true, y_pred, axis_name: str, path: Path, title: str):
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=6, alpha=0.4)
    lo = min(np.nanmin(y_true), np.nanmin(y_pred))
    hi = max(np.nanmax(y_true), np.nanmax(y_pred))
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel(f"{axis_name}_true (mm)")
    plt.ylabel(f"{axis_name}_pred (mm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def plot_scatter_summary(y_test, preds, outdir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        ax = axes[axis_idx]
        ax.scatter(y_test[:, axis_idx], preds[:, axis_idx], s=5, alpha=0.35)
        lo = min(np.nanmin(y_test[:, axis_idx]), np.nanmin(preds[:, axis_idx]))
        hi = max(np.nanmax(y_test[:, axis_idx]), np.nanmax(preds[:, axis_idx]))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"{axis_name}_true (mm)")
        ax.set_ylabel(f"{axis_name}_pred (mm)")
        ax.set_title(axis_name)
    fig.tight_layout()
    fig.savefig(outdir / "scatter.png")
    plt.close(fig)

    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        plot_scatter_1d(
            y_test[:, axis_idx],
            preds[:, axis_idx],
            axis_name,
            outdir / f"scatter-{axis_name}.png",
            f"{axis_name}_pred vs {axis_name}_true (test set)",
        )


def plot_pull(resid, path: Path, title: str, xlabel: str, mae=None, rmse=None):
    plt.figure(figsize=(6, 4))
    counts, edges, _ = plt.hist(resid, bins=40, alpha=0.85)
    plt.xlabel(xlabel)
    plt.ylabel("Counts")
    plt.title(title)

    centers = 0.5 * (edges[:-1] + edges[1:])

    def _gauss(x, A, mu, sigma):
        return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

    A0 = counts.max() if len(counts) > 0 else 1.0
    mu0 = float(np.mean(resid)) if len(resid) > 0 else 0.0
    sigma0 = float(np.std(resid)) if len(resid) > 0 else 1.0
    resid_std = float(np.std(resid)) if len(resid) > 0 else np.nan

    fit_ok = False
    mu_err = np.nan
    try:
        from scipy.optimize import curve_fit
        if len(resid) < 5 or sigma0 <= 0:
            raise RuntimeError("Not enough residuals for a stable Gaussian fit")
        popt, pcov = curve_fit(_gauss, centers, counts, p0=[A0, mu0, sigma0], maxfev=20000)
        A, mu, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        mu_err = perr[1] if len(perr) > 1 else np.nan
        fit_ok = True
    except Exception:
        A, mu, sigma = A0, mu0, sigma0

    sigma = abs(float(sigma)) if sigma is not None else np.nan
    fwhm = 2.355 * sigma if np.isfinite(sigma) else np.nan
    xfit = np.linspace(edges[0], edges[-1], 200) if len(edges) > 1 else np.array([0.0, 1.0])
    yfit = _gauss(xfit, A, mu, sigma) if sigma is not None and sigma != 0 else np.zeros_like(xfit)
    plt.plot(xfit, yfit, "r-", label="Gaussian fit" if fit_ok else "Gaussian (moment est.)")

    txt = (
        f"Data std = {resid_std:.3g}\n"
        f"$\\mu$ = {mu:.3g} ± {mu_err:.2g}\n"
        f"$\\sigma$ = {sigma:.3g}\n"
        f"FWHM = {fwhm:.3g}"
    )
    if mae is not None and rmse is not None:
        txt += f"\nMAE = {mae:.3g}\nRMSE = {rmse:.3g}"
    plt.text(
        0.98, 0.95, txt, transform=plt.gca().transAxes,
        ha="right", va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8), fontsize=9
    )
    plt.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

    return {
        "fit_ok": bool(fit_ok),
        "resid_std_mm": float(resid_std),
        "fit_mu_mm": float(mu),
        "fit_mu_err_mm": float(mu_err),
        "fit_sigma_mm": float(sigma),
        "fit_fwhm_mm": float(fwhm),
    }


def assign_true_pixel(test_meta: pd.DataFrame):
    if "x_gamma" not in test_meta.columns or "y_gamma" not in test_meta.columns:
        raise RuntimeError("Per-pixel diagnostics require x_gamma and y_gamma columns in the CSV")

    x = test_meta["x_gamma"].to_numpy(dtype=float)
    y = test_meta["y_gamma"].to_numpy(dtype=float)

    ix = np.rint((x + PIXEL_OFFSET_MM) / PIXEL_PITCH_MM).astype(int)
    iy = np.rint((y + PIXEL_OFFSET_MM) / PIXEL_PITCH_MM).astype(int)

    cx = ix * PIXEL_PITCH_MM - PIXEL_OFFSET_MM
    cy = iy * PIXEL_PITCH_MM - PIXEL_OFFSET_MM

    inside_pixel = (
        (ix >= 0) & (ix < IMAGE_SIZE) &
        (iy >= 0) & (iy < IMAGE_SIZE) &
        (np.abs(x - cx) <= 0.5 * PIXEL_SIZE_MM) &
        (np.abs(y - cy) <= 0.5 * PIXEL_SIZE_MM)
    )
    return ix, iy, inside_pixel


def plot_doi_resolution_map(metrics: dict, path: Path):
    fwhm_map = np.full((IMAGE_SIZE, IMAGE_SIZE), np.nan)
    n_map = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=int)

    for px in range(IMAGE_SIZE):
        for py in range(IMAGE_SIZE):
            key = f"pixel_{px}_{py}"
            pixel_metrics = metrics.get(key, {})
            fwhm_map[py, px] = pixel_metrics.get("fit_fwhm_mm", np.nan)
            n_map[py, px] = int(pixel_metrics.get("n_test", 0))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fwhm_map, origin="lower", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DOI FWHM resolution (mm)")

    ax.set_xticks(np.arange(IMAGE_SIZE))
    ax.set_yticks(np.arange(IMAGE_SIZE))
    ax.set_xlabel("LYSO pixel x index")
    ax.set_ylabel("LYSO pixel y index")
    ax.set_title("Per-pixel DOI resolution from z residual Gaussian FWHM")

    for py in range(IMAGE_SIZE):
        for px in range(IMAGE_SIZE):
            value = fwhm_map[py, px]
            if np.isfinite(value):
                text = f"{value:.2f}\n(n={n_map[py, px]})"
            else:
                text = f"n={n_map[py, px]}"
            ax.text(px, py, text, ha="center", va="center", color="white", fontsize=7)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_per_pixel_doi_diagnostics(y_test, preds, test_meta: pd.DataFrame, outdir: Path):
    pixel_dir = outdir / "per-pixel"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    try:
        ix, iy, inside_pixel = assign_true_pixel(test_meta)
    except RuntimeError as exc:
        print(f"Skipping per-pixel diagnostics: {exc}")
        return {}

    z_true = y_test[:, 2]
    z_pred = preds[:, 2]
    metrics = {}
    skipped = 0

    for px in range(IMAGE_SIZE):
        for py in range(IMAGE_SIZE):
            mask = (ix == px) & (iy == py) & inside_pixel
            n = int(mask.sum())
            key = f"pixel_{px}_{py}"
            metrics[key] = {"n_test": n}
            if n < MIN_EVENTS_PER_PIXEL_PLOT:
                skipped += 1
                continue

            y_sel = z_true[mask]
            p_sel = z_pred[mask]
            resid = y_sel - p_sel
            mae = float(np.mean(np.abs(resid)))
            rmse = float(np.sqrt(np.mean(resid ** 2)))
            metrics[key].update({"z_mae_mm": mae, "z_rmse_mm": rmse})

            label = f"pixel ({px}, {py}), n={n}"
            plot_scatter_1d(
                y_sel,
                p_sel,
                "z",
                pixel_dir / f"scatter_z_pixel_{px}_{py}.png",
                f"z_pred vs z_true: {label}",
            )
            fit_metrics = plot_pull(
                resid,
                pixel_dir / f"pull_z_pixel_{px}_{py}.png",
                f"z residuals: {label}",
                "z_true - z_pred (mm)",
                mae=mae,
                rmse=rmse,
            )
            metrics[key].update(fit_metrics)

    plot_doi_resolution_map(metrics, pixel_dir / "doi-resolution-fwhm-map.png")
    if skipped:
        print(f"Skipped {skipped} per-pixel plot pairs with fewer than {MIN_EVENTS_PER_PIXEL_PLOT} test events")
    print(f"Wrote per-pixel DOI diagnostics to {pixel_dir}")
    return metrics


def evaluate(y_test, preds, outdir: Path, test_meta: pd.DataFrame | None = None):
    residual = y_test - preds
    component_mae = np.mean(np.abs(residual), axis=0)
    component_rmse = np.sqrt(np.mean(residual ** 2, axis=0))
    error_3d = np.linalg.norm(residual, axis=1)

    print(
        "Test mean 3D error: "
        f"{np.mean(error_3d):.4f} mm | median: {np.median(error_3d):.4f} mm"
    )
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        print(
            f"  {axis_name}: MAE={component_mae[axis_idx]:.4f} mm | "
            f"RMSE={component_rmse[axis_idx]:.4f} mm"
        )

    plot_scatter_summary(y_test, preds, outdir)

    pull_metrics = {}
    for axis_idx, axis_name in enumerate(AXIS_NAMES):
        pull_metrics[axis_name] = plot_pull(
            residual[:, axis_idx],
            outdir / f"pull-{axis_name}.png",
            f"{axis_name} residuals (test set)",
            f"{axis_name}_true - {axis_name}_pred (mm)",
            mae=float(component_mae[axis_idx]),
            rmse=float(component_rmse[axis_idx]),
        )

    plt.figure(figsize=(6, 4))
    plt.hist(error_3d, bins=50, alpha=0.85)
    plt.xlabel("3D position error |r_true - r_pred| (mm)")
    plt.ylabel("Counts")
    plt.title("3D position error (test set)")
    plt.tight_layout()
    plt.savefig(outdir / "position-error-3d.png")
    plt.close()

    per_pixel_metrics = {}
    if test_meta is not None:
        per_pixel_metrics = plot_per_pixel_doi_diagnostics(y_test, preds, test_meta, outdir)

    metrics = {
        "mean_3d_error_mm": float(np.mean(error_3d)),
        "median_3d_error_mm": float(np.median(error_3d)),
        "p68_3d_error_mm": float(np.percentile(error_3d, 68)),
        "p95_3d_error_mm": float(np.percentile(error_3d, 95)),
        "component_mae_mm": {a: float(component_mae[i]) for i, a in enumerate(AXIS_NAMES)},
        "component_rmse_mm": {a: float(component_rmse[i]) for i, a in enumerate(AXIS_NAMES)},
        "pull": pull_metrics,
        "per_pixel": per_pixel_metrics,
    }
    return metrics


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    bounds_np = np.array(
        [
            [args.x_min, args.x_max],
            [args.y_min, args.y_max],
            [args.z_min, args.z_max],
        ],
        dtype=np.float32,
    )
    oob_relative_weights_np = np.array(
        [
            args.oob_relative_weight if args.x_oob_relative_weight is None else args.x_oob_relative_weight,
            args.oob_relative_weight if args.y_oob_relative_weight is None else args.y_oob_relative_weight,
            args.oob_relative_weight if args.z_oob_relative_weight is None else args.z_oob_relative_weight,
        ],
        dtype=np.float32,
    )

    print(f"Target columns: {list(args.target_columns)}")
    print(f"Bounds [x, y, z] mm: {bounds_np.tolist()}")
    print(f"OOB relative weights [x, y, z]: {oob_relative_weights_np.tolist()}")

    set_reproducible(args.seed)
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = prep_data(
        csv_path=args.csv,
        seed=args.seed,
        normalize=args.normalize,
        target_columns=args.target_columns,
        test_mode=args.test_mode,
        test_mode_events=args.test_mode_events,
    )
    y_test, preds, history, best_val = train_cnn(
        X_train, X_val, X_test, y_train, y_val, y_test,
        outdir=args.outdir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        bounds_np=bounds_np,
        oob_relative_weights_np=oob_relative_weights_np,
    )
    metrics = evaluate(y_test, preds, args.outdir, test_meta=test_meta)
    metrics["best_val_loss"] = float(best_val)
    metrics["best_val_mean_3d_error_mm"] = float(min(history["val_distance"]))
    metrics["bounds_mm"] = {
        a: [float(bounds_np[i, 0]), float(bounds_np[i, 1])]
        for i, a in enumerate(AXIS_NAMES)
    }
    metrics["oob_relative_weights"] = {
        a: float(oob_relative_weights_np[i])
        for i, a in enumerate(AXIS_NAMES)
    }

    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
