#imports
import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from evaluate_cnn_performance import evaluate_model_performance

###################################################
# Local variable definitions
IMAGE_SIZE = 8
PIXEL_SIZE_MM = 3.0
PIXEL_GAP_MM = 0.2
PIXEL_PITCH_MM = PIXEL_SIZE_MM + PIXEL_GAP_MM
PIXEL_OFFSET_MM = 0.5 * (IMAGE_SIZE - 1) * PIXEL_PITCH_MM #mid-point of the array in mm (with the mid of first pixel as origin?)
XY_HALF_WIDTH_MM = PIXEL_OFFSET_MM + 0.5 * PIXEL_SIZE_MM # half the x or y size of the array
MIN_EVENTS_PER_PIXEL_PLOT = 5

RANDOM_SEED = 1234
BATCH_SIZE = 256
EPOCHS = 100
LR = 3e-4
NORMALIZE = "none"  # "sum", "max", "log1p", or "none"
TARGET_COLUMNS = ("x_gamma", "y_gamma", "z_gamma")
OUTPUT_NAMES = ("x", "y", "z")
TOTAL_PE_WINDOW = (None, None)
SIGMA_MIN_MM = 0.05
SIGMA_MAX_MM = 50.0
Z_GMM_COMPONENTS = 2

X_MIN = -XY_HALF_WIDTH_MM
X_MAX = XY_HALF_WIDTH_MM
Y_MIN = -XY_HALF_WIDTH_MM
Y_MAX = XY_HALF_WIDTH_MM
Z_MIN = -7.5
Z_MAX = 7.5

OUT_OF_BOUNDS_RELATIVE_WEIGHT = 0.0
LOSS_EPS = 1e-12
###################################################

# Parse arguments for configurable variable definitions
def optional_float(value):
    if isinstance(value, str) and value.lower() in {"none", "null"}:
        return None
    return float(value)


def format_optional_float(value):
    return "None" if value is None else f"{value:g}"


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
    parser.add_argument("--total-pe-min", type=optional_float, default=TOTAL_PE_WINDOW[0], help="Lower total_pe selection; use None to disable")
    parser.add_argument("--total-pe-max", type=optional_float, default=TOTAL_PE_WINDOW[1], help="Upper total_pe selection; use None to disable")
    parser.add_argument("--test-mode", action="store_true", help="Use a small random subset for quick debugging")
    parser.add_argument("--test-mode-events", type=int, default=2000, help="Maximum rows used when --test-mode is set")
    parser.add_argument("--x-min", type=float, default=X_MIN)
    parser.add_argument("--x-max", type=float, default=X_MAX)
    parser.add_argument("--y-min", type=float, default=Y_MIN)
    parser.add_argument("--y-max", type=float, default=Y_MAX)
    parser.add_argument("--z-min", type=float, default=Z_MIN)
    parser.add_argument("--z-max", type=float, default=Z_MAX)
    parser.add_argument("--sigma-min-mm", type=float, default=SIGMA_MIN_MM)
    parser.add_argument("--sigma-max-mm", type=float, default=SIGMA_MAX_MM)
    parser.add_argument("--z-gmm-components", type=int, default=Z_GMM_COMPONENTS)
    parser.add_argument("--oob-relative-weight", type=float, default=OUT_OF_BOUNDS_RELATIVE_WEIGHT)
    parser.add_argument("--x-oob-relative-weight", type=float, default=None)
    parser.add_argument("--y-oob-relative-weight", type=float, default=None)
    parser.add_argument("--z-oob-relative-weight", type=float, default=None)
    return parser.parse_args()

# setting seeds for reproducibiliyy
def set_reproducible(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(1)

def prep_data(
    csv_path: Path,
    seed: int,
    normalize: str,
    target_columns,
    test_mode: bool,
    test_mode_events: int,
    total_pe_min,
    total_pe_max,
):
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    pe_cols = [c for c in df.columns if c.startswith("pe_")]
    if len(pe_cols) == 0:
        raise RuntimeError("No pe_ columns found in CSV")

    missing_targets = [c for c in target_columns if c not in df.columns]
    if missing_targets:
        raise RuntimeError(f"Missing target columns: {missing_targets}")

    use_total_pe_cut = total_pe_min is not None or total_pe_max is not None
    if use_total_pe_cut and "total_pe" not in df.columns:
        raise RuntimeError("Missing total_pe column needed for the PE window cut")

    # removing non-finite values, if any
    valid = np.ones(len(df), dtype=bool)
    for col in target_columns:
        valid &= np.isfinite(df[col].to_numpy(dtype=float))
    if not np.all(valid):
        dropped = int((~valid).sum())
        print(f"Dropping {dropped} rows with non-finite target values")
        df = df.loc[valid].reset_index(drop=True)

    if use_total_pe_cut:
        total_pe = df["total_pe"].to_numpy(dtype=float)
        pe_window = np.isfinite(total_pe)
        if total_pe_min is not None:
            pe_window &= total_pe >= total_pe_min
        if total_pe_max is not None:
            pe_window &= total_pe <= total_pe_max

        if not np.all(pe_window):
            dropped = int((~pe_window).sum())
            print(
                "Applying total_pe window "
                f"[{format_optional_float(total_pe_min)}, {format_optional_float(total_pe_max)}]: "
                f"keeping {int(pe_window.sum())}, dropping {dropped}"
            )
            df = df.loc[pe_window].reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("No rows remain after applying the total_pe window")
    else:
        print("Skipping total_pe window cut")

    #construct 'energy maps'
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

    #define target vectors
    y = df[list(target_columns)].to_numpy(dtype=np.float32)

    #split into training/test/validation
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
        df.iloc[idx_test].reset_index(drop=True), #return full dataframe for test-set events (useful for post-training analysis)
    )

# define CNN architecture
class SimpleCNN(nn.Module):
    def __init__(self, in_channels: int, out_dim: int = 4 + 3 * Z_GMM_COMPONENTS):
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
            nn.Linear(8 * 3 * 2 * 2, 64),# 3 conv branches, 8 output channels from each conv, conv output 2 x 2 
            nn.ReLU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        b2 = self.branch_k2(x)
        b4 = self.branch_k4(x)
        b8 = self.branch_k8(x)
        feats = torch.cat([b2, b4, b8], dim=1)
        return self.head(feats)

def xy_zgmm_output_dim(z_components: int):
    # x/y each have mean and log-sigma. z has mixture logits, means, and log-sigmas.
    return 4 + 3 * z_components




def split_xy_zgmm_output(raw_pred, log_sigma_min, log_sigma_max, z_components: int):
    if z_components < 1:
        raise ValueError("z_components must be >= 1")

    xy_mean = raw_pred[:, 0:2]
    xy_log_sigma = torch.clamp(raw_pred[:, 2:4], min=log_sigma_min, max=log_sigma_max)
    xy_sigma = torch.exp(xy_log_sigma)

    i0 = 4
    i1 = i0 + z_components
    i2 = i1 + z_components
    i3 = i2 + z_components
    if raw_pred.shape[1] != i3:
        raise ValueError(f"Expected {i3} network outputs, got {raw_pred.shape[1]}")

    z_logits = raw_pred[:, i0:i1]
    z_mu = raw_pred[:, i1:i2]
    z_log_sigma = torch.clamp(raw_pred[:, i2:i3], min=log_sigma_min, max=log_sigma_max)
    z_sigma = torch.exp(z_log_sigma)
    z_weight = torch.softmax(z_logits, dim=1)
    z_log_weight = torch.log_softmax(z_logits, dim=1)

    z_mean = (z_weight * z_mu).sum(dim=1)
    z_second_moment = (z_weight * (z_sigma ** 2 + z_mu ** 2)).sum(dim=1)
    z_var = (z_second_moment - z_mean ** 2).clamp_min(LOSS_EPS)
    z_effective_sigma = torch.sqrt(z_var)

    mean = torch.cat([xy_mean, z_mean[:, None]], dim=1)
    sigma = torch.cat([xy_sigma, z_effective_sigma[:, None]], dim=1)
    z_params = {
        "weight": z_weight,
        "log_weight": z_log_weight,
        "mu": z_mu,
        "sigma": z_sigma,
        "log_sigma": z_log_sigma,
    }
    return mean, sigma, xy_log_sigma, z_params


def xy_zgmm_nll_terms(raw_pred, target, log_sigma_min, log_sigma_max, z_components: int):
    mean, sigma, xy_log_sigma, z_params = split_xy_zgmm_output(
        raw_pred,
        log_sigma_min,
        log_sigma_max,
        z_components,
    )

    xy_residual = mean[:, 0:2] - target[:, 0:2]
    xy_nll_terms = 0.5 * (xy_residual / sigma[:, 0:2]) ** 2 + xy_log_sigma

    z_true = target[:, 2:3]
    z_log_prob = (
        -0.5 * ((z_true - z_params["mu"]) / z_params["sigma"]) ** 2
        - z_params["log_sigma"]
    )
    z_nll = -torch.logsumexp(z_params["log_weight"] + z_log_prob, dim=1)

    # Average over the three target coordinates so the scale stays close to
    # the original heteroscedastic Gaussian loss.
    nll = (xy_nll_terms.sum(dim=1) + z_nll).mean() / len(OUTPUT_NAMES)
    spatial_distance = torch.linalg.vector_norm(mean - target, dim=1).mean()
    return nll, spatial_distance, mean, sigma, z_params

def xyz_loss_terms(pred, target):
    spatial_distance = torch.linalg.vector_norm(pred[:, :3] - target[:, :3], dim=1).mean()
    return spatial_distance


def oob_penalty_terms_xyz(pred, base_loss, bounds, relative_weights):
    weighted_total = pred.new_tensor(0.0)
    raw_terms = {}
    weighted_terms = {}

    for idx, name in enumerate(OUTPUT_NAMES):
        lo = bounds[idx, 0]
        hi = bounds[idx, 1]
        rel_weight = float(relative_weights[idx])

        raw_penalty = (
            torch.relu(pred[:, idx] - hi) ** 2
            + torch.relu(lo - pred[:, idx]) ** 2
        ).mean()

        if rel_weight <= 0:
            weighted_penalty = pred.new_tensor(0.0)
        else:
            scale = base_loss.detach() / raw_penalty.detach().clamp_min(LOSS_EPS)
            weighted_penalty = rel_weight * scale * raw_penalty

        raw_terms[name] = raw_penalty
        weighted_terms[name] = weighted_penalty
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
    sigma_min_mm: float,
    sigma_max_mm: float,
    z_gmm_components: int,
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
    log_sigma_min = float(np.log(max(sigma_min_mm, LOSS_EPS)))
    log_sigma_max = float(np.log(max(sigma_max_mm, sigma_min_mm + LOSS_EPS)))

    model = SimpleCNN(X_train.shape[1], out_dim=xy_zgmm_output_dim(z_gmm_components)).to(device)
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
        "train_nll": [],
        "train_spatial_distance": [],
        "train_oob_weighted": [],
        "train_total": [],
        "val_nll": [],
        "val_spatial_distance": [],
        "val_oob_weighted": [],
        "val_total": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        train_nll = 0.0
        train_spatial = 0.0
        train_oob = 0.0
        train_total = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad()
            raw_pred = model(xb)
            nll, spatial_distance, mean_pred, _, _ = xy_zgmm_nll_terms(
                raw_pred,
                yb,
                log_sigma_min,
                log_sigma_max,
                z_gmm_components,
            )
            _, _, oob_total = oob_penalty_terms_xyz(
                mean_pred, spatial_distance, bounds, oob_relative_weights
            )
            loss = nll + oob_total
            loss.backward()
            opt.step()

            n = xb.size(0)
            train_nll += nll.item() * n
            train_spatial += spatial_distance.item() * n
            train_oob += oob_total.item() * n
            train_total += loss.item() * n

        train_nll /= len(train_ds)
        train_spatial /= len(train_ds)
        train_oob /= len(train_ds)
        train_total /= len(train_ds)

        model.eval()
        val_nll = 0.0
        val_spatial = 0.0
        val_oob = 0.0
        val_total = 0.0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                raw_pred = model(xb)
                nll, spatial_distance, mean_pred, _, _ = xy_zgmm_nll_terms(
                    raw_pred,
                    yb,
                    log_sigma_min,
                    log_sigma_max,
                    z_gmm_components,
                )
                _, _, oob_total = oob_penalty_terms_xyz(
                    mean_pred, spatial_distance, bounds, oob_relative_weights
                )
                loss = nll + oob_total

                n = xb.size(0)
                val_nll += nll.item() * n
                val_spatial += spatial_distance.item() * n
                val_oob += oob_total.item() * n
                val_total += loss.item() * n

        val_nll /= len(val_ds)
        val_spatial /= len(val_ds)
        val_oob /= len(val_ds)
        val_total /= len(val_ds)

        print(
            f"Epoch {epoch:02d} | "
            f"train NLL={train_nll:.4f} | "
            f"train D3={train_spatial:.4f} | "
            f"train OOB={train_oob:.4f} | "
            f"val NLL={val_nll:.4f} | "
            f"val D3={val_spatial:.4f} | "
            f"val OOB={val_oob:.4f}"
        )

        history["epoch"].append(epoch)
        history["train_nll"].append(train_nll)
        history["train_spatial_distance"].append(train_spatial)
        history["train_oob_weighted"].append(train_oob)
        history["train_total"].append(train_total)
        history["val_nll"].append(val_nll)
        history["val_spatial_distance"].append(val_spatial)
        history["val_oob_weighted"].append(val_oob)
        history["val_total"].append(val_total)

        if val_total < best_val:
            best_val = val_total
            torch.save(model.state_dict(), best_model_path)

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_nll"], label="Train NLL")
    plt.plot(history["epoch"], history["val_nll"], label="Val NLL")
    plt.plot(history["epoch"], history["train_total"], "--", label="Train total")
    plt.plot(history["epoch"], history["val_total"], "--", label="Val total")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training-curves.png")
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(history["epoch"], history["train_spatial_distance"], label="Train mean 3D error")
    plt.plot(history["epoch"], history["val_spatial_distance"], label="Val mean 3D error")
    plt.xlabel("Epoch")
    plt.ylabel("3D error (mm)")
    plt.title("Position error during training")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "training-curves-3d-error.png")
    plt.close()

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    preds = []
    pred_sigmas = []
    z_gmm_weights = []
    z_gmm_mus = []
    z_gmm_sigmas = []
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            raw_pred = model(xb)
            mean_pred, sigma_pred, _, z_params = split_xy_zgmm_output(
                raw_pred,
                log_sigma_min,
                log_sigma_max,
                z_gmm_components,
            )
            preds.append(mean_pred.cpu().numpy())
            pred_sigmas.append(sigma_pred.cpu().numpy())
            z_gmm_weights.append(z_params["weight"].cpu().numpy())
            z_gmm_mus.append(z_params["mu"].cpu().numpy())
            z_gmm_sigmas.append(z_params["sigma"].cpu().numpy())

    preds = np.concatenate(preds)
    pred_sigmas = np.concatenate(pred_sigmas)
    z_gmm_predictions = {
        "weight": np.concatenate(z_gmm_weights),
        "mu": np.concatenate(z_gmm_mus),
        "sigma": np.concatenate(z_gmm_sigmas),
    }
    return y_test, preds, pred_sigmas, z_gmm_predictions, history, best_val


def main():
    args = parse_args()
    if args.z_gmm_components < 1:
        raise ValueError("--z-gmm-components must be >= 1")
    args.outdir.mkdir(parents=True, exist_ok=True)

    set_reproducible(args.seed)
    X_train, X_val, X_test, y_train, y_val, y_test, test_meta = prep_data(
        csv_path=args.csv,
        seed=args.seed,
        normalize=args.normalize,
        target_columns=args.target_columns,
        test_mode=args.test_mode,
        test_mode_events=args.test_mode_events,
        total_pe_min=args.total_pe_min,
        total_pe_max=args.total_pe_max,
    )

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
    print(f"total_pe window: [{format_optional_float(args.total_pe_min)}, {format_optional_float(args.total_pe_max)}]")
    print(f"Bounds [x, y, z]: {bounds_np.tolist()}")
    print(f"OOB relative weights [x, y, z]: {oob_relative_weights_np.tolist()}")
    print(f"Predicted sigma bounds: [{args.sigma_min_mm:g}, {args.sigma_max_mm:g}] mm")
    print(f"DOI z likelihood: {args.z_gmm_components}-component Gaussian mixture")
    print("Uncertainty diagnostic cut: sigma below the per-axis median")

    y_test, preds, pred_sigmas, z_gmm_predictions, history, best_val = train_cnn(
        X_train, X_val, X_test, y_train, y_val, y_test,
        outdir=args.outdir,
        seed=args.seed,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        bounds_np=bounds_np,
        oob_relative_weights_np=oob_relative_weights_np,
        sigma_min_mm=args.sigma_min_mm,
        sigma_max_mm=args.sigma_max_mm,
        z_gmm_components=args.z_gmm_components,
    )

    preds_for_eval = preds.copy()
    z_gmm_mean = preds[:, 2]
    print("Using z GMM mean as the z point prediction for evaluation")

    metrics = evaluate_model_performance(
        y_true=y_test,
        y_pred=preds_for_eval,
        outdir=args.outdir,
        output_names=OUTPUT_NAMES,
        spatial_indices=(0, 1, 2),
        predicted_uncertainties=pred_sigmas,
        metadata=test_meta,
        process_column="process",
        pixel_geometry={
            "image_size": IMAGE_SIZE,
            "pixel_size_mm": PIXEL_SIZE_MM,
            "pitch_mm": PIXEL_PITCH_MM,
            "offset_mm": PIXEL_OFFSET_MM,
        },
        min_events_per_pixel=MIN_EVENTS_PER_PIXEL_PLOT,
    )
    metrics["best_val_loss"] = float(best_val)
    metrics["best_val_mean_3d_error_mm"] = float(min(history["val_spatial_distance"]))
    metrics["bounds"] = {name: [float(bounds_np[i, 0]), float(bounds_np[i, 1])] for i, name in enumerate(OUTPUT_NAMES)}
    metrics["oob_relative_weights"] = {name: float(oob_relative_weights_np[i]) for i, name in enumerate(OUTPUT_NAMES)}
    metrics["total_pe_window"] = [args.total_pe_min, args.total_pe_max]
    metrics["sigma_bounds_mm"] = [float(args.sigma_min_mm), float(args.sigma_max_mm)]
    metrics["z_gmm_components"] = int(args.z_gmm_components)
    metrics["z_point_prediction"] = "gmm_mean"

    pred_df = test_meta.copy()
    for idx, name in enumerate(OUTPUT_NAMES):
        pred_df[f"{name}_true"] = y_test[:, idx]
        pred_df[f"{name}_pred"] = preds_for_eval[:, idx]
        pred_df[f"{name}_sigma"] = pred_sigmas[:, idx]
    pred_df["z_gmm_mean"] = z_gmm_mean
    for component in range(args.z_gmm_components):
        pred_df[f"z_gmm_weight_{component}"] = z_gmm_predictions["weight"][:, component]
        pred_df[f"z_gmm_mu_{component}"] = z_gmm_predictions["mu"][:, component]
        pred_df[f"z_gmm_sigma_{component}"] = z_gmm_predictions["sigma"][:, component]
    pred_df.to_csv(args.outdir / "test-predictions-with-uncertainty.csv", index=False)

    with open(args.outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
