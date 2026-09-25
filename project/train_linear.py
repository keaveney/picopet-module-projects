import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

CSV_PATH = Path("df_training.csv")
OUTPUT_DIR = Path("linear-performance")

# Match notebook defaults
XLIM_Z = (-7.5, 7.5)  # set to None for auto
N_Z_BINS = 6

def _weighted_linear_fit(x, y, yerr):
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr) & (yerr > 0)
    if mask.sum() < 2:
        return np.nan, np.nan, mask
    w = 1.0 / (yerr[mask] ** 2)
    x = x[mask]
    y = y[mask]
    S = np.sum(w)
    Sx = np.sum(w * x)
    Sy = np.sum(w * y)
    Sxx = np.sum(w * x * x)
    Sxy = np.sum(w * x * y)
    den = S * Sxx - Sx * Sx
    if den == 0:
        return np.nan, np.nan, mask
    m = (S * Sxy - Sx * Sy) / den
    b = (Sy - m * Sx) / S
    return m, b, mask

def _gauss(x, A, mu, sigma):
    return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def train_linear(csv_path=CSV_PATH,
                 output_dir=OUTPUT_DIR,
                 xlim_z=XLIM_Z,
                 n_z_bins=N_Z_BINS):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required = {"z_gamma", "frac_max", "max_pe"}
    if not required.issubset(df.columns):
        missing = sorted(required - set(df.columns))
        raise RuntimeError(f"Missing columns in CSV: {missing}")

    plot_df = df.copy()

    if len(plot_df) == 0:
        raise RuntimeError("No events left after applying filters.")

    if xlim_z is None:
        zmin, zmax = float(plot_df["z_gamma"].min()), float(plot_df["z_gamma"].max())
    else:
        zmin, zmax = xlim_z

    if zmin == zmax:
        raise RuntimeError("Z range is zero; cannot bin for linear fit.")

    bins = np.linspace(zmin, zmax, n_z_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    # 1D distributions of max pixel fraction for Z bins
    plt.figure(figsize=(7, 4))
    for i in range(n_z_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (plot_df["z_gamma"] >= lo) & (plot_df["z_gamma"] < hi)
        if mask.sum() == 0:
            continue
        plt.hist(
            plot_df.loc[mask, "frac_max"],
            bins=np.linspace(0.58, 0.77, 50),
            #bins=np.linspace(0.2, 1.0, 50),
            density=True,
            histtype="bar",
            alpha=0.35,
            edgecolor="k",
            label=f"{lo:.2f} to {hi:.2f} mm",
        )

    plt.xlabel("Max pixel fraction")
    plt.ylabel("Density")
    plt.title("Max pixel fraction by Z bin")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "frac_max_by_z_bin.png")
    plt.close()

    means = []
    sems = []
    for i in range(n_z_bins):
        lo, hi = bins[i], bins[i + 1]
        vals = plot_df.loc[(plot_df["z_gamma"] >= lo) & (plot_df["z_gamma"] < hi), "frac_max"]
        if len(vals) == 0:
            means.append(np.nan)
            sems.append(np.nan)
        else:
            means.append(vals.mean())
            sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else np.nan)

    centers_arr = np.array(centers)
    means_arr = np.array(means)
    sems_arr = np.array(sems)

    m, b, mask_fit = _weighted_linear_fit(centers_arr, means_arr, sems_arr)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot: mean frac_max vs Z bin center with linear fit
    plt.figure(figsize=(6, 4))
    plt.errorbar(centers_arr, means_arr, yerr=sems_arr, fmt='o', capsize=3, label='Means')
    if np.isfinite(m) and np.isfinite(b):
        xfit = np.linspace(centers_arr[mask_fit].min(), centers_arr[mask_fit].max(), 100)
        yfit = m * xfit + b
        plt.plot(xfit, yfit, '-', label=f'Fit: y = {m:.3g} x + {b:.3g}')
    plt.xlabel('Z bin center (mm)')
    plt.ylabel('Mean max pixel fraction')
    plt.title('Linear fit: mean max pixel fraction vs Z')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "linear-fit.png")
    plt.close()

    if not np.isfinite(m) or m == 0:
        raise RuntimeError("Fit slope is invalid; cannot invert for Z.")

    z_est = (plot_df["frac_max"] - b) / m
    dz = plot_df["z_gamma"] - z_est

   # Scatter: Z_pred vs Z_true (test set)
    plt.figure(figsize=(5, 5))
    plt.scatter(plot_df["z_gamma"], z_est, s=6, alpha=0.4)
    minv = float(min(plot_df["z_gamma"].min(), z_est.min()))
    maxv = float(max(plot_df["z_gamma"].max(), z_est.max()))
    #plt.plot([-7.5, 7.5], [minv, maxv], 'r--', linewidth=1)
    plt.xlabel("Z_true (mm)")
    plt.ylabel("Z_pred (mm)")
    plt.title("Z_pred vs Z_true (test set)")
    plt.tight_layout()
    plt.savefig("linear-performance/scatter.png")


    # Residuals histogram + Gaussian fit
    plt.figure(figsize=(6, 4))
    counts, edges, _ = plt.hist(dz, bins=50, alpha=0.85)
    plt.xlabel('Z_true - Z_est (mm)')
    plt.ylabel('Counts')
    plt.title('Z residuals from linear frac_max->Z inversion')

    centers = 0.5 * (edges[:-1] + edges[1:])
    A0 = counts.max() if len(counts) > 0 else 1.0
    mu0 = float(np.mean(dz)) if len(dz) > 0 else 0.0
    sigma0 = float(np.std(dz)) if len(dz) > 0 else 1.0
    resid_std = float(np.std(dz)) if len(dz) > 0 else np.nan

    mu_err = np.nan
    fit_ok = False
    try:
        from scipy.optimize import curve_fit
        popt, pcov = curve_fit(_gauss, centers, counts, p0=[A0, mu0, sigma0], maxfev=20000)
        A, mu, sigma = popt
        perr = np.sqrt(np.diag(pcov))
        mu_err = perr[1] if len(perr) > 1 else np.nan
        fit_ok = True
    except Exception:
        A, mu, sigma = A0, mu0, sigma0
        if len(dz) > 0 and sigma0 > 0:
            mu_err = sigma0 / np.sqrt(len(dz))

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
    plt.savefig(output_dir / "linear-residuals.png")
    plt.close()

    return {
        "slope": float(m),
        "intercept": float(b),
        "fwhm": float(fwhm),
        "mu": float(mu),
        "sigma": float(sigma),
        "data_std": float(resid_std),
        "n_events": int(len(plot_df)),
        "output_dir": str(output_dir),
    }


if __name__ == "__main__":
    result = train_linear()
    print("Linear fit result:")
    for k, v in result.items():
        print(f"  {k}: {v}")
