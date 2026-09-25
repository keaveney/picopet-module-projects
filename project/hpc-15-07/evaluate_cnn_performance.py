import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from html_report import write_html_report
except ImportError:
    from hpc.html_report import write_html_report

LOSS_EPS = 1e-12
FULL_ENERGY_DELTA_KE = 0.511
VISIBLE_ENERGY_CALIBRATION_KEV = 1000.0 * FULL_ENERGY_DELTA_KE
PHOTOPEAK_KEV = 511.0
COMPTON_EDGE_KEV = 340.7
KLEIN_NISHINA_MAXIMUM_KEV = 180
FULL_ENERGY_DELTA_KE_ATOL = 1e-4
FULL_ENERGY_DELTA_KE_RTOL = 1e-3
PE_SPECTRUM_MODE_MIN = 100.0
PE_SPECTRUM_FIT_FRACTION = 0.35
PAPER_FONT_SIZE = 14
PAPER_LABEL_SIZE = 16
PAPER_TICK_SIZE = 13
PAPER_LEGEND_SIZE = 11
PAPER_TEXT_SIZE = 11
PAPER_SMALL_TEXT_SIZE = 10
VALIDATION_FIGSIZE = (7.2, 5.0)
VALIDATION_SPECTRUM_FIGSIZE = (7.6, 5.2)
OUTPUT_NAMES = ("x", "y", "z")
SPATIAL_INDICES = (0, 1, 2)
PIXEL_GEOMETRY = {
    "image_size": 8,
    "pixel_size_mm": 3.0,
    "pitch_mm": 3.2,
    "offset_mm": 11.2,
    "pixel_length_mm": 15.0,
}
SOURCE_RADIUS_MM = 13.0
SOURCE_DISTANCE_CM = 20.0
CONE_MARGIN = 0.3
CONE_HALF_ANGLE_DEG = None
CONE_ANGLE_SAMPLES = 96
MIN_EVENTS_PER_PIXEL = 5
PE_SPECTRUM_FIT_MODEL = "gauss-linear"
N_PRED_Z_CONDITION_GMM_COMPONENTS = 4
Z_GMM_MODE_GRID_SIZE = 801

plt.rcParams.update(
    {
        "font.size": PAPER_FONT_SIZE,
        "axes.labelsize": PAPER_LABEL_SIZE,
        "xtick.labelsize": PAPER_TICK_SIZE,
        "ytick.labelsize": PAPER_TICK_SIZE,
        "legend.fontsize": PAPER_LEGEND_SIZE,
        "figure.titlesize": PAPER_LABEL_SIZE,
    }
)


def _as_path(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _style_axis_labels(ax):
    """Use paper-style axis labels without subplot titles."""
    ax.set_title("")
    if ax.get_xlabel():
        ax.xaxis.set_label_coords(1.0, -0.10)
        ax.xaxis.label.set_horizontalalignment("right")
    if ax.get_ylabel():
        ax.yaxis.set_label_coords(-0.17, 1.0)
        ax.yaxis.label.set_verticalalignment("top")


def _style_figure(fig):
    for ax in fig.axes:
        _style_axis_labels(ax)


def _save_paper_figure(fig, path, dpi=160):
    _style_figure(fig)
    fig.tight_layout(pad=1.05)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)



def _extract_z_gmm_arrays(metadata):
    if metadata is None:
        return None

    weights = []
    means = []
    sigmas = []
    component = 0
    while True:
        w_col = f"z_gmm_weight_{component}"
        m_col = f"z_gmm_mu_{component}"
        s_col = f"z_gmm_sigma_{component}"
        if not all(col in metadata.columns for col in (w_col, m_col, s_col)):
            break
        weights.append(metadata[w_col].to_numpy(dtype=float))
        means.append(metadata[m_col].to_numpy(dtype=float))
        sigmas.append(metadata[s_col].to_numpy(dtype=float))
        component += 1

    if component == 0:
        return None

    return np.stack(weights, axis=1), np.stack(means, axis=1), np.stack(sigmas, axis=1)


def use_z_gmm_mean_prediction(y_pred, output_names, metadata):
    """Return a copy of y_pred with z replaced by the ordinary GMM mean."""
    if "z" not in output_names or metadata is None:
        return y_pred, False

    z_index = output_names.index("z")
    y_pred_mean = np.array(y_pred, copy=True)

    if "z_gmm_mean" in metadata.columns:
        y_pred_mean[:, z_index] = metadata["z_gmm_mean"].to_numpy(dtype=float)
        return y_pred_mean, True

    z_gmm_arrays = _extract_z_gmm_arrays(metadata)
    if z_gmm_arrays is None:
        return y_pred, False

    weights, means, _ = z_gmm_arrays
    weights = np.maximum(weights, LOSS_EPS)
    weights = weights / weights.sum(axis=1, keepdims=True)
    y_pred_mean[:, z_index] = (weights * means).sum(axis=1)
    return y_pred_mean, True


def _normal_cdf_values(x):
    x = np.asarray(x, dtype=float)
    try:
        from scipy.special import ndtr

        return ndtr(x)
    except Exception:
        erf = np.vectorize(math.erf)
        return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _normal_pdf_values(x, mean, sigma):
    sigma = np.maximum(np.asarray(sigma, dtype=float), LOSS_EPS)
    return np.exp(-0.5 * ((np.asarray(x)[..., None] - mean) / sigma) ** 2) / (
        np.sqrt(2.0 * np.pi) * sigma
    )


def _gmm_median_predictions(weights, means, sigmas, z_bounds, iterations=40, chunk_size=50000):
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), LOSS_EPS)
    weights = np.maximum(weights, LOSS_EPS)
    weights = weights / weights.sum(axis=1, keepdims=True)

    n_events = weights.shape[0]
    result = np.full(n_events, np.nan, dtype=float)
    physical_lo, physical_hi = z_bounds

    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        w = weights[start:stop]
        mu = means[start:stop]
        sigma = sigmas[start:stop]

        lo = np.minimum(np.min(mu - 8.0 * sigma, axis=1), physical_lo)
        hi = np.maximum(np.max(mu + 8.0 * sigma, axis=1), physical_hi)

        for _ in range(iterations):
            mid = 0.5 * (lo + hi)
            cdf = np.sum(w * _normal_cdf_values((mid[:, None] - mu) / sigma), axis=1)
            lo = np.where(cdf < 0.5, mid, lo)
            hi = np.where(cdf >= 0.5, mid, hi)

        result[start:stop] = 0.5 * (lo + hi)

    return result


def _gmm_mode_predictions(weights, means, sigmas, z_bounds, grid_size=801, chunk_size=5000):
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), LOSS_EPS)
    weights = np.maximum(weights, LOSS_EPS)
    weights = weights / weights.sum(axis=1, keepdims=True)

    n_events = weights.shape[0]
    z_grid = np.linspace(float(z_bounds[0]), float(z_bounds[1]), int(max(grid_size, 25)))
    result = np.full(n_events, np.nan, dtype=float)

    for start in range(0, n_events, chunk_size):
        stop = min(start + chunk_size, n_events)
        w = weights[start:stop]
        mu = means[start:stop]
        sigma = sigmas[start:stop]
        pdf = np.zeros((len(z_grid), stop - start), dtype=float)
        for component in range(w.shape[1]):
            pdf += (
                w[None, :, component]
                * np.exp(-0.5 * ((z_grid[:, None] - mu[None, :, component]) / sigma[None, :, component]) ** 2)
                / (np.sqrt(2.0 * np.pi) * sigma[None, :, component])
            )
        result[start:stop] = z_grid[np.argmax(pdf, axis=0)]

    return result


def use_z_gmm_point_prediction(
    y_pred,
    output_names,
    metadata,
    estimator="mean",
    z_bounds=None,
    mode_grid_size=801,
):
    """Return y_pred with z replaced by a selected GMM point estimator."""
    estimator = str(estimator).lower()
    if estimator == "input":
        return y_pred, False
    if estimator == "mean":
        y_pred_mean, used = use_z_gmm_mean_prediction(y_pred, output_names, metadata)
        return y_pred_mean, used

    if "z" not in output_names or metadata is None:
        return y_pred, False

    z_index = output_names.index("z")
    z_gmm_arrays = _extract_z_gmm_arrays(metadata)
    if z_gmm_arrays is None:
        return y_pred, False

    weights, means, sigmas = z_gmm_arrays
    if z_bounds is None:
        finite_means = means[np.isfinite(means)]
        finite_sigmas = sigmas[np.isfinite(sigmas)]
        if len(finite_means) == 0:
            return y_pred, False
        pad = 8.0 * float(np.nanmedian(finite_sigmas)) if len(finite_sigmas) > 0 else 1.0
        z_bounds = (float(np.nanmin(finite_means) - pad), float(np.nanmax(finite_means) + pad))

    y_pred_estimator = np.array(y_pred, copy=True)
    if estimator == "median":
        y_pred_estimator[:, z_index] = _gmm_median_predictions(weights, means, sigmas, z_bounds)
    elif estimator == "mode":
        y_pred_estimator[:, z_index] = _gmm_mode_predictions(
            weights,
            means,
            sigmas,
            z_bounds,
            grid_size=mode_grid_size,
        )
    else:
        raise ValueError(f"Unknown z GMM point estimator: {estimator}")

    return y_pred_estimator, True


def _derive_3x3_pe_observables(metadata, image_size=8):
    pe_cols = []
    for col in metadata.columns:
        if not col.startswith("pe_"):
            continue
        parts = col.split("_")
        if len(parts) != 3:
            continue
        try:
            ix = int(parts[1])
            iy = int(parts[2])
        except ValueError:
            continue
        if 0 <= ix < image_size and 0 <= iy < image_size:
            pe_cols.append((iy * image_size + ix, ix, iy, col))

    if len(pe_cols) != image_size * image_size:
        return {}

    pe_cols = sorted(pe_cols)
    col_names = [item[3] for item in pe_cols]
    pe = metadata[col_names].to_numpy(dtype=float)
    max_flat = np.nanargmax(pe, axis=1)
    pe_3x3 = np.zeros(len(metadata), dtype=float)

    neighbour_indices = []
    for flat in range(image_size * image_size):
        ix = flat % image_size
        iy = flat // image_size
        neighbours = []
        for jy in range(max(0, iy - 1), min(image_size, iy + 2)):
            for jx in range(max(0, ix - 1), min(image_size, ix + 2)):
                neighbours.append(jy * image_size + jx)
        neighbour_indices.append(neighbours)

    for flat, neighbours in enumerate(neighbour_indices):
        mask = max_flat == flat
        if np.any(mask):
            pe_3x3[mask] = pe[mask][:, neighbours].sum(axis=1)

    result = {"pe_3x3_around_max": pe_3x3}
    if "total_pe" in metadata.columns:
        total_pe = metadata["total_pe"].to_numpy(dtype=float)
        result["frac_3x3_around_max"] = np.divide(
            pe_3x3,
            total_pe,
            out=np.full_like(pe_3x3, np.nan, dtype=float),
            where=total_pe > 0,
        )
    return result


def plot_doi_observables_vs_true_z(y_true, metadata, outdir, output_names, spatial_indices=(0, 1, 2)):
    if metadata is None or "z" not in output_names:
        return {}

    z_index = output_names.index("z")
    z_true = np.asarray(y_true)[:, z_index]

    candidate_columns = [
        ("total_pe", "Total PE"),
        ("max_pe", "Max pixel PE"),
        ("frac_max", "Max pixel fraction"),
        ("spread", "Light spread"),
        ("pe_3x3_around_max", "3x3 PE around max pixel"),
        ("frac_3x3_around_max", "3x3 fraction around max pixel"),
    ]

    derived = _derive_3x3_pe_observables(metadata)
    observables = []
    for col, label in candidate_columns:
        if col in metadata.columns:
            values = metadata[col].to_numpy(dtype=float)
        elif col in derived:
            values = derived[col]
        else:
            continue
        observables.append((col, label, values))

    if not observables:
        return {}

    n_cols = 2
    n_rows = int(np.ceil(len(observables) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.2 * n_cols, 4.3 * n_rows), squeeze=False)
    axes_flat = axes.ravel()
    rng = np.random.default_rng(12345)
    metrics = {}

    z_bins = np.linspace(np.nanmin(z_true), np.nanmax(z_true), 16)
    z_centres = 0.5 * (z_bins[:-1] + z_bins[1:])

    for ax, (col, label, values) in zip(axes_flat, observables):
        finite = np.isfinite(z_true) & np.isfinite(values)
        z = z_true[finite]
        obs = values[finite]

        if len(z) == 0:
            ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        max_scatter = 50000
        if len(z) > max_scatter:
            draw = rng.choice(len(z), size=max_scatter, replace=False)
        else:
            draw = np.arange(len(z))
        ax.scatter(z[draw], obs[draw], s=4, alpha=0.12, color="tab:blue", rasterized=True)

        medians = []
        q16 = []
        q84 = []
        counts = []
        for lo, hi in zip(z_bins[:-1], z_bins[1:]):
            mask = (z >= lo) & (z < hi)
            counts.append(int(mask.sum()))
            if mask.sum() >= 10:
                medians.append(float(np.median(obs[mask])))
                q16.append(float(np.percentile(obs[mask], 16)))
                q84.append(float(np.percentile(obs[mask], 84)))
            else:
                medians.append(np.nan)
                q16.append(np.nan)
                q84.append(np.nan)

        medians = np.asarray(medians)
        q16 = np.asarray(q16)
        q84 = np.asarray(q84)
        good = np.isfinite(medians)
        if np.any(good):
            ax.plot(z_centres[good], medians[good], color="black", lw=1.8, label="median")
            ax.fill_between(z_centres[good], q16[good], q84[good], color="black", alpha=0.16, label="16-84%")

        corr = np.corrcoef(z, obs)[0, 1] if len(z) > 1 and np.std(obs) > 0 and np.std(z) > 0 else np.nan
        rank_corr = pd.Series(z).rank().corr(pd.Series(obs).rank()) if len(z) > 1 else np.nan
        metrics[col] = {
            "n": int(len(z)),
            "pearson_corr_with_z_true": float(corr) if np.isfinite(corr) else np.nan,
            "spearman_corr_with_z_true": float(rank_corr) if np.isfinite(rank_corr) else np.nan,
            "z_bin_counts": counts,
        }

        ax.set_xlabel("true z (mm)", loc="right")
        ax.set_ylabel(label, loc="top")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=PAPER_SMALL_TEXT_SIZE, loc="best")

    for ax in axes_flat[len(observables):]:
        ax.set_axis_off()

    _save_paper_figure(fig, outdir / "doi-observables-vs-true-z.png")
    return metrics

def _gaussian(x, amplitude, mean, sigma):
    sigma = np.maximum(np.abs(sigma), LOSS_EPS)
    return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _fit_gaussian_to_histogram(values, bins=40):
    counts, edges = np.histogram(values, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    amplitude0 = counts.max() if len(counts) > 0 else 1.0
    mean0 = float(np.mean(values)) if len(values) > 0 else 0.0
    sigma0 = float(np.std(values)) if len(values) > 0 else 1.0

    fit_ok = False
    mean_err = np.nan
    try:
        from scipy.optimize import curve_fit

        if len(values) < 5 or sigma0 <= 0:
            raise RuntimeError("Not enough residuals for a stable Gaussian fit")
        popt, pcov = curve_fit(
            _gaussian,
            centers,
            counts,
            p0=[amplitude0, mean0, sigma0],
            maxfev=20000,
        )
        perr = np.sqrt(np.diag(pcov))
        mean_err = perr[1] if len(perr) > 1 else np.nan
        fit_ok = True
    except Exception:
        popt = np.asarray([amplitude0, mean0, sigma0], dtype=float)

    amplitude, mean, sigma = popt
    sigma = abs(float(sigma))
    return {
        "counts": counts,
        "edges": edges,
        "fit_ok": bool(fit_ok),
        "amplitude": float(amplitude),
        "mean": float(mean),
        "mean_err": float(mean_err),
        "sigma": float(sigma),
        "fwhm": float(2.355 * sigma),
        "data_std": float(np.std(values)) if len(values) > 0 else np.nan,
    }


def _fit_gaussian_pdf(values, bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 5:
        return {
            "fit_ok": False,
            "amplitude": np.nan,
            "mean": np.nan,
            "sigma": np.nan,
            "fwhm": np.nan,
        }

    counts, edges = np.histogram(values, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    finite = np.isfinite(counts) & (counts > 0)

    amplitude0 = float(np.nanmax(counts)) if np.any(finite) else 1.0
    mean0 = float(np.mean(values))
    sigma0 = float(np.std(values, ddof=1)) if len(values) > 1 else 1.0
    sigma0 = max(sigma0, LOSS_EPS)

    fit_ok = False
    try:
        from scipy.optimize import curve_fit

        if np.count_nonzero(finite) < 5:
            raise RuntimeError("Not enough populated bins for Gaussian fit")
        popt, _ = curve_fit(
            _gaussian,
            centers[finite],
            counts[finite],
            p0=[amplitude0, mean0, sigma0],
            maxfev=20000,
        )
        fit_ok = True
    except Exception:
        popt = np.asarray([amplitude0, mean0, sigma0], dtype=float)

    amplitude, mean, sigma = popt
    sigma = abs(float(sigma))
    return {
        "fit_ok": bool(fit_ok),
        "amplitude": float(amplitude),
        "mean": float(mean),
        "sigma": sigma,
        "fwhm": float(2.355 * sigma),
    }


def _truncated_exponential_nll(log_mu, depth_mm, length_mm, cos_theta):
    mu = np.exp(log_mu)
    cos_theta = np.clip(np.asarray(cos_theta, dtype=float), 1e-6, 1.0)
    lam = mu / cos_theta
    tau = lam * length_mm
    log_norm = np.log1p(-np.exp(-tau))
    log_pdf = np.log(lam) - lam * depth_mm - log_norm
    return -float(np.sum(log_pdf))


def _truncated_exponential_log_pdf(depth_mm, length_mm, cos_theta, mu):
    depth_mm = np.asarray(depth_mm, dtype=float)
    cos_theta = np.clip(np.asarray(cos_theta, dtype=float), 1e-6, 1.0)
    lam = mu / cos_theta
    tau = lam * length_mm
    log_norm = np.log1p(-np.exp(-tau))
    return np.log(lam) - lam * depth_mm - log_norm


def _cone_averaged_truncated_exponential_nll(log_mu, depth_mm, length_mm, cos_samples):
    """NLL for depth with incidence angle marginalized over a uniform cone."""
    mu = np.exp(log_mu)
    depth_mm = np.asarray(depth_mm, dtype=float)
    cos_samples = np.clip(np.asarray(cos_samples, dtype=float), 1e-6, 1.0)
    total = 0.0
    chunk_size = 20000

    for start in range(0, len(depth_mm), chunk_size):
        d = depth_mm[start:start + chunk_size, None]
        log_pdf = _truncated_exponential_log_pdf(d, length_mm, cos_samples[None, :], mu)
        max_log_pdf = np.max(log_pdf, axis=1)
        log_mean_pdf = max_log_pdf + np.log(np.mean(np.exp(log_pdf - max_log_pdf[:, None]), axis=1))
        total -= float(np.sum(log_mean_pdf))

    return total


def _fit_cone_averaged_truncated_exponential_mu(depth_mm, length_mm, cos_samples):
    depth_mm = np.asarray(depth_mm, dtype=float)
    depth_mm = depth_mm[np.isfinite(depth_mm)]
    cos_samples = np.asarray(cos_samples, dtype=float)
    cos_samples = cos_samples[np.isfinite(cos_samples) & (cos_samples > 0)]

    if len(depth_mm) < 5 or length_mm <= 0 or len(cos_samples) < 2:
        return {
            "fit_ok": False,
            "mu_per_mm": np.nan,
            "attenuation_length_mm": np.nan,
            "reason": "too few values, invalid length, or invalid cone samples",
        }

    fit_ok = False
    reason = "ok"
    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            _cone_averaged_truncated_exponential_nll,
            args=(depth_mm, length_mm, cos_samples),
            bounds=(-16.0, 4.0),
            method="bounded",
            options={"xatol": 1e-8},
        )
        mu = float(np.exp(result.x))
        fit_ok = bool(result.success)
        if not fit_ok:
            reason = str(result.message)
    except Exception as exc:
        log_grid = np.linspace(-16.0, 4.0, 600)
        nll = np.asarray([
            _cone_averaged_truncated_exponential_nll(v, depth_mm, length_mm, cos_samples)
            for v in log_grid
        ])
        best = int(np.nanargmin(nll))
        mu = float(np.exp(log_grid[best]))
        fit_ok = True
        reason = f"grid fallback: {exc}"

    return {
        "fit_ok": bool(fit_ok),
        "mu_per_mm": float(mu),
        "attenuation_length_mm": float(1.0 / mu) if mu > 0 else np.inf,
        "reason": reason,
    }


def _fit_truncated_exponential_mu(depth_mm, length_mm, cos_theta=None):
    """Fit positive mu for an angle-averaged truncated first-interaction model.

    For event i with incidence angle theta_i, the z-depth density is

        p(d_i | theta_i, interaction) =
            (mu / cos(theta_i)) exp[-mu d_i / cos(theta_i)]
            / [1 - exp(-mu L / cos(theta_i))]

    with 0 < d_i < L. If cos_theta is omitted, normal incidence is assumed.
    """
    depth_mm = np.asarray(depth_mm, dtype=float)
    if cos_theta is None:
        cos_theta = np.ones_like(depth_mm)
    else:
        cos_theta = np.asarray(cos_theta, dtype=float)

    finite = np.isfinite(depth_mm) & np.isfinite(cos_theta) & (cos_theta > 0)
    depth_mm = depth_mm[finite]
    cos_theta = np.clip(cos_theta[finite], 1e-6, 1.0)

    if len(depth_mm) < 5 or length_mm <= 0:
        return {
            "fit_ok": False,
            "mu_per_mm": np.nan,
            "attenuation_length_mm": np.nan,
            "reason": "too few values or invalid length",
        }

    fit_ok = False
    reason = "ok"
    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            _truncated_exponential_nll,
            args=(depth_mm, length_mm, cos_theta),
            bounds=(-16.0, 4.0),
            method="bounded",
            options={"xatol": 1e-8},
        )
        mu = float(np.exp(result.x))
        fit_ok = bool(result.success)
        if not fit_ok:
            reason = str(result.message)
    except Exception as exc:
        # Lightweight fallback for environments without scipy.
        log_grid = np.linspace(-16.0, 4.0, 600)
        nll = np.asarray([
            _truncated_exponential_nll(v, depth_mm, length_mm, cos_theta)
            for v in log_grid
        ])
        best = int(np.nanargmin(nll))
        mu = float(np.exp(log_grid[best]))
        fit_ok = True
        reason = f"grid fallback: {exc}"

    return {
        "fit_ok": bool(fit_ok),
        "mu_per_mm": float(mu),
        "attenuation_length_mm": float(1.0 / mu) if mu > 0 else np.inf,
        "reason": reason,
    }


def _angle_averaged_truncated_exponential_counts(edges, mu, length_mm, cos_theta):
    cos_theta = np.clip(np.asarray(cos_theta, dtype=float), 1e-6, 1.0)
    if mu <= 0:
        return len(cos_theta) * np.diff(edges) / length_mm

    lam = mu / cos_theta[:, None]
    lo = edges[:-1][None, :]
    hi = edges[1:][None, :]
    norm = 1.0 - np.exp(-lam * length_mm)
    probabilities = (np.exp(-lam * lo) - np.exp(-lam * hi)) / np.maximum(norm, LOSS_EPS)
    return probabilities.sum(axis=0)


def _cone_averaged_truncated_exponential_counts(edges, mu, length_mm, cos_samples, n_events):
    cos_samples = np.clip(np.asarray(cos_samples, dtype=float), 1e-6, 1.0)
    if mu <= 0:
        return n_events * np.diff(edges) / length_mm

    lam = mu / cos_samples[:, None]
    lo = edges[:-1][None, :]
    hi = edges[1:][None, :]
    norm = 1.0 - np.exp(-lam * length_mm)
    probabilities = (np.exp(-lam * lo) - np.exp(-lam * hi)) / np.maximum(norm, LOSS_EPS)
    return n_events * np.mean(probabilities, axis=0)


def _cone_averaged_truncated_exponential_binned_nll(log_mu, counts, edges, length_mm, cos_samples):
    """Fast binned NLL for the cone-averaged attenuation model."""
    counts = np.asarray(counts, dtype=float)
    n_events = float(np.sum(counts))
    if n_events <= 0:
        return np.inf

    expected = _cone_averaged_truncated_exponential_counts(
        edges,
        float(np.exp(log_mu)),
        length_mm,
        cos_samples,
        n_events=1.0,
    )
    probabilities = expected / np.maximum(np.sum(expected), LOSS_EPS)
    return -float(np.sum(counts * np.log(np.maximum(probabilities, LOSS_EPS))))


def _fit_cone_averaged_truncated_exponential_binned_mu(counts, edges, length_mm, cos_samples):
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    cos_samples = np.asarray(cos_samples, dtype=float)
    cos_samples = cos_samples[np.isfinite(cos_samples) & (cos_samples > 0)]

    if np.sum(counts) < 5 or length_mm <= 0 or len(cos_samples) < 2:
        return {
            "fit_ok": False,
            "mu_per_mm": np.nan,
            "attenuation_length_mm": np.nan,
            "reason": "too few values, invalid length, or invalid cone samples",
            "fit_type": "binned",
        }

    fit_ok = False
    reason = "ok"
    try:
        from scipy.optimize import minimize_scalar

        result = minimize_scalar(
            _cone_averaged_truncated_exponential_binned_nll,
            args=(counts, edges, length_mm, cos_samples),
            bounds=(-16.0, 4.0),
            method="bounded",
            options={"xatol": 1e-8},
        )
        mu = float(np.exp(result.x))
        fit_ok = bool(result.success)
        if not fit_ok:
            reason = str(result.message)
    except Exception as exc:
        log_grid = np.linspace(-16.0, 4.0, 600)
        nll = np.asarray([
            _cone_averaged_truncated_exponential_binned_nll(v, counts, edges, length_mm, cos_samples)
            for v in log_grid
        ])
        best = int(np.nanargmin(nll))
        mu = float(np.exp(log_grid[best]))
        fit_ok = True
        reason = f"grid fallback: {exc}"

    return {
        "fit_ok": bool(fit_ok),
        "mu_per_mm": float(mu),
        "attenuation_length_mm": float(1.0 / mu) if mu > 0 else np.inf,
        "reason": reason,
        "fit_type": "binned",
    }


def _cone_cos_samples(
    pixel_geometry,
    source_radius_mm=13.0,
    source_distance_cm=20.0,
    cone_margin=0.3,
    cone_half_angle_deg=None,
    n_samples=96,
):
    if cone_half_angle_deg is not None:
        theta_max = np.deg2rad(float(cone_half_angle_deg))
    else:
        if pixel_geometry is None:
            return None, np.nan
        image_size = float(pixel_geometry.get("image_size", 8))
        pixel_size_mm = float(pixel_geometry.get("pixel_size_mm", 3.0))
        pitch_mm = float(pixel_geometry.get("pitch_mm", 3.2))
        gap_mm = max(pitch_mm - pixel_size_mm, 0.0)
        block_xy_mm = image_size * pixel_size_mm + (image_size + 1.0) * gap_mm
        source_distance_mm = 10.0 * float(source_distance_cm)
        acceptance_radius_mm = float(cone_margin) * (0.5 * block_xy_mm + float(source_radius_mm))
        theta_max = np.arctan2(acceptance_radius_mm, source_distance_mm)

    theta_max = max(theta_max, 0.0)
    cos_min = np.cos(theta_max)
    n_samples = max(int(n_samples), 2)
    u = (np.arange(n_samples, dtype=float) + 0.5) / n_samples
    cos_samples = cos_min + u * (1.0 - cos_min)
    return cos_samples, float(np.rad2deg(theta_max))


def plot_gamma_first_interaction_depth_validation(
    y_true,
    metadata,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    pixel_geometry=None,
    n_bins=40,
    min_events=50,
    incident_energy_window_mev=0.001,
    source_radius_mm=13.0,
    source_distance_cm=20.0,
    cone_margin=0.3,
    cone_half_angle_deg=None,
    cone_angle_samples=96,
):
    """Validate gamma transport with first-interaction depth vs attenuation fit."""
    if "z" in output_names:
        z_index = output_names.index("z")
    else:
        z_index = spatial_indices[2]

    if metadata is not None and "z_gamma" in metadata.columns:
        z_mm = metadata["z_gamma"].to_numpy(dtype=float)
        z_source = "z_gamma"
    else:
        z_mm = np.asarray(y_true)[:, z_index]
        z_source = f"{output_names[z_index]}_true"

    if pixel_geometry is not None and "pixel_length_mm" in pixel_geometry:
        pixel_length_mm = float(pixel_geometry["pixel_length_mm"])
        z_front_mm = -0.5 * pixel_length_mm
    else:
        finite_z = z_mm[np.isfinite(z_mm)]
        if len(finite_z) < min_events:
            print("Skipping gamma depth validation: too few finite z values")
            return {}
        z_front_mm = float(np.nanmin(finite_z))
        pixel_length_mm = float(np.nanmax(finite_z) - z_front_mm)

    depth_mm = z_mm - z_front_mm
    valid = np.isfinite(depth_mm) & (depth_mm >= 0.0) & (depth_mm <= pixel_length_mm)

    energy_selection = "all finite first interactions"
    n_energy_window = None
    if metadata is not None and "gamma_incident_energy" in metadata.columns:
        energy = metadata["gamma_incident_energy"].to_numpy(dtype=float)
        finite_energy = np.isfinite(energy)
        if np.any(finite_energy):
            # OpenGATE-derived files in this project normally store MeV, but
            # older diagnostics may store keV. Infer from scale.
            energy_mev = energy / 1000.0 if np.nanmedian(energy[finite_energy]) > 10.0 else energy
            energy_mask = finite_energy & (np.abs(energy_mev - FULL_ENERGY_DELTA_KE) <= incident_energy_window_mev)
            n_energy_window = int(np.sum(valid & energy_mask))
            if n_energy_window >= min_events:
                valid &= energy_mask
                energy_selection = (
                    rf"$|E_\gamma - 511\,\mathrm{{keV}}| < "
                    rf"{1000.0 * incident_energy_window_mev:.0f}\,\mathrm{{keV}}$"
                )

    depth = depth_mm[valid]
    depth = depth[np.isfinite(depth)]
    if len(depth) < min_events:
        print(f"Skipping gamma depth validation: only {len(depth)} selected interactions")
        return {
            "n_selected": int(len(depth)),
            "z_source": z_source,
            "energy_selection": energy_selection,
            "n_energy_window": n_energy_window,
        }

    bins = np.linspace(0.0, pixel_length_mm, n_bins + 1)
    counts, edges = np.histogram(depth, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    cos_samples, theta_max_deg = _cone_cos_samples(
        pixel_geometry,
        source_radius_mm=source_radius_mm,
        source_distance_cm=source_distance_cm,
        cone_margin=cone_margin,
        cone_half_angle_deg=cone_half_angle_deg,
        n_samples=cone_angle_samples,
    )
    if cos_samples is None:
        cos_samples = np.ones(int(cone_angle_samples), dtype=float)
        theta_max_deg = 0.0
        angle_model = "normal-incidence fallback"
    else:
        angle_model = "uniform solid-angle cone average"

    fit = _fit_cone_averaged_truncated_exponential_binned_mu(
        counts,
        edges,
        pixel_length_mm,
        cos_samples,
    )
    expected = _cone_averaged_truncated_exponential_counts(
        edges,
        fit["mu_per_mm"],
        pixel_length_mm,
        cos_samples,
        len(depth),
    )
    expected_norm = expected / np.sum(expected) if np.sum(expected) > 0 else expected

    fig, ax = plt.subplots(figsize=VALIDATION_FIGSIZE)
    ax.hist(
        depth,
        bins=bins,
        weights=np.full(len(depth), 1.0 / len(depth)),
        histtype="stepfilled",
        alpha=0.78,
        color="tab:blue",
        label="simulation",
    )
    ax.step(
        centers,
        expected_norm,
        where="mid",
        color="tab:red",
        lw=2.0,
        label="attenuation model",
    )
    ax.set_xlabel("First interaction z (mm)", loc="right")
    ax.set_ylabel("Fraction of events / bin", loc="top")
    ax.legend(fontsize=PAPER_SMALL_TEXT_SIZE, loc="best")

    attenuation = fit["attenuation_length_mm"]
    if np.isfinite(attenuation):
        attenuation_text = rf"fitted $\lambda={attenuation:.1f}\,\mathrm{{mm}}$"
    else:
        attenuation_text = r"fitted $\lambda\to\infty$"
    ax.text(
        0.97,
        0.94,
        attenuation_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        fontsize=PAPER_TEXT_SIZE,
    )

    _save_paper_figure(fig, outdir / "gamma-first-interaction-depth-attenuation.png")

    return {
        "n_selected": int(len(depth)),
        "z_source": z_source,
        "z_front_mm": float(z_front_mm),
        "pixel_length_mm": float(pixel_length_mm),
        "energy_selection": energy_selection,
        "n_energy_window": n_energy_window,
        "angle_model": angle_model,
        "cone_half_angle_deg": float(theta_max_deg),
        "source_radius_mm": float(source_radius_mm),
        "source_distance_cm": float(source_distance_cm),
        "cone_margin": float(cone_margin),
        "mean_cos_theta": float(np.mean(cos_samples)),
        "min_cos_theta": float(np.min(cos_samples)),
        "max_cos_theta": float(np.max(cos_samples)),
        "mean_depth_mm": float(np.mean(depth)),
        "fit_ok": bool(fit["fit_ok"]),
        "fit_reason": fit["reason"],
        "mu_per_mm": float(fit["mu_per_mm"]),
        "attenuation_length_mm": float(fit["attenuation_length_mm"]),
    }


def plot_gamma_incident_angle_distribution(
    metadata,
    outdir,
    min_events=10,
):
    if metadata is None or "gamma_incident_angle_deg" not in metadata.columns:
        print("Skipping incident-angle distribution: missing gamma_incident_angle_deg")
        return {}

    angle = metadata["gamma_incident_angle_deg"].to_numpy(dtype=float)
    finite = np.isfinite(angle)
    if finite.sum() < min_events:
        print("Skipping incident-angle distribution: too few finite angles")
        return {"n": int(finite.sum())}

    angle = angle[finite]
    lo = 0.0
    hi = max(float(np.nanpercentile(angle, 99.8)), float(np.nanmax(angle)), 1.0)
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=VALIDATION_FIGSIZE)
    ax.hist(angle, bins=bins, alpha=0.82, color="tab:blue")
    ax.set_xlabel("Incident gamma angle to pixel normal (deg)", loc="right")
    ax.set_ylabel("Events / bin", loc="top")
    ax.text(
        0.97,
        0.94,
        (
            f"n = {len(angle)}\n"
            rf"$\langle \theta \rangle = {np.mean(angle):.3g}^\circ$"
            "\n"
            rf"median $\theta = {np.median(angle):.3g}^\circ$"
            "\n"
            rf"95% $< {np.percentile(angle, 95):.3g}^\circ$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        fontsize=PAPER_SMALL_TEXT_SIZE,
    )
    _save_paper_figure(fig, outdir / "gamma-incident-angle-distribution.png")

    return {
        "n": int(len(angle)),
        "mean_deg": float(np.mean(angle)),
        "median_deg": float(np.median(angle)),
        "std_deg": float(np.std(angle, ddof=1)) if len(angle) > 1 else np.nan,
        "p95_deg": float(np.percentile(angle, 95)),
        "max_deg": float(np.max(angle)),
    }


def plot_prediction_scatter(y_true, y_pred, output_names, outdir):
    n_outputs = len(output_names)
    ncols = 2 if n_outputs <= 4 else 3
    nrows = int(np.ceil(n_outputs / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows))
    axes = np.asarray(axes).reshape(-1)

    for idx, name in enumerate(output_names):
        ax = axes[idx]
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=5, alpha=0.35)
        lo = min(np.nanmin(y_true[:, idx]), np.nanmin(y_pred[:, idx]))
        hi = max(np.nanmax(y_true[:, idx]), np.nanmax(y_pred[:, idx]))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1)
        ax.set_xlabel(f"{name}_true", loc="right")
        ax.set_ylabel(f"{name}_pred", loc="top")

    for ax in axes[n_outputs:]:
        ax.axis("off")

    _save_paper_figure(fig, outdir / "scatter.png")


def plot_residual_histogram(residual, path, title, xlabel):
    fit = _fit_gaussian_to_histogram(residual, bins=40)
    counts = fit["counts"]
    edges = fit["edges"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residual, bins=edges, alpha=0.85)
    ax.set_xlabel(xlabel, loc="right")
    ax.set_ylabel("Counts", loc="top")

    xfit = np.linspace(edges[0], edges[-1], 200) if len(edges) > 1 else np.array([0.0, 1.0])
    yfit = _gaussian(xfit, fit["amplitude"], fit["mean"], fit["sigma"])
    ax.plot(xfit, yfit, "r-", label="Gaussian fit" if fit["fit_ok"] else "Gaussian (moment est.)")

    txt = (
        f"Data std = {fit['data_std']:.3g}\n"
        f"$\\mu$ = {fit['mean']:.3g} +/- {fit['mean_err']:.2g}\n"
        f"$\\sigma$ = {fit['sigma']:.3g}\n"
        f"FWHM = {fit['fwhm']:.3g}"
    )
    ax.text(
        0.98,
        0.95,
        txt,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        fontsize=9,
    )
    ax.legend(fontsize=8, loc="best")
    _save_paper_figure(fig, path)

    return {
        "fit_ok": fit["fit_ok"],
        "resid_std": fit["data_std"],
        "fit_mu": fit["mean"],
        "fit_mu_err": fit["mean_err"],
        "fit_sigma": fit["sigma"],
        "fit_fwhm": fit["fwhm"],
    }


def _residual_summary(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "rmse": np.nan,
            "fwhm_est": np.nan,
        }

    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "std": std,
        "rmse": float(np.sqrt(np.mean(values ** 2))),
        "fwhm_est": float(2.355 * std),
    }


def plot_spatial_pulls_by_gamma_delta_ke(
    y_true,
    y_pred,
    metadata,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    delta_ke_column="gamma_delta_ke",
    full_energy_value=FULL_ENERGY_DELTA_KE,
):
    """Compare x/y/z residuals for full-energy PE events against all other events."""
    if metadata is None or delta_ke_column not in metadata.columns:
        print(f"Skipping gamma_delta_ke pull comparison: missing {delta_ke_column}")
        return {}

    delta_ke = metadata[delta_ke_column].to_numpy(dtype=float)
    residual = np.asarray(y_true) - np.asarray(y_pred)

    full_energy_mask = np.isclose(
        delta_ke,
        full_energy_value,
        atol=FULL_ENERGY_DELTA_KE_ATOL,
        rtol=FULL_ENERGY_DELTA_KE_RTOL,
    )
    valid = np.isfinite(delta_ke) & np.all(np.isfinite(residual[:, list(spatial_indices)]), axis=1)
    full_energy_mask &= valid
    other_mask = valid & ~full_energy_mask

    n_full = int(full_energy_mask.sum())
    n_other = int(other_mask.sum())
    if n_full == 0 or n_other == 0:
        print(
            "Skipping gamma_delta_ke pull comparison: "
            f"n_full_energy={n_full}, n_other={n_other}"
        )
        return {
            "n_full_energy": n_full,
            "n_other": n_other,
            "full_energy_value": float(full_energy_value),
        }

    fig, axes = plt.subplots(1, len(spatial_indices), figsize=(5.2 * len(spatial_indices), 4.2))
    axes = np.asarray(axes).reshape(-1)

    metrics = {
        "full_energy_value": float(full_energy_value),
        "n_full_energy": n_full,
        "n_other": n_other,
        "axes": {},
    }

    for ax, idx in zip(axes, spatial_indices):
        name = output_names[idx]
        full_residual = residual[full_energy_mask, idx]
        other_residual = residual[other_mask, idx]
        combined = np.concatenate([full_residual, other_residual])
        lo, hi = np.nanpercentile(combined, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
        if lo == hi:
            lo -= 0.5
            hi += 0.5

        bins = np.linspace(lo, hi, 45)
        ax.hist(
            other_residual,
            bins=bins,
            density=True,
            alpha=0.45,
            label="Other first interactions",
            color="tab:orange",
        )
        ax.hist(
            full_residual,
            bins=bins,
            density=True,
            alpha=0.45,
            label=f"{delta_ke_column} = {full_energy_value:g}",
            color="tab:blue",
        )
        ax.axvline(0.0, color="black", ls="--", lw=1)
        ax.set_xlabel(f"{name}_true - {name}_pred (mm)", loc="right")
        ax.set_ylabel("Density", loc="top")

        full_stats = _residual_summary(full_residual)
        other_stats = _residual_summary(other_residual)
        metrics["axes"][name] = {
            "full_energy": full_stats,
            "other": other_stats,
        }

        txt = (
            f"Full-energy PE: n={full_stats['n']}\n"
            f"  mean={full_stats['mean']:.3g} mm, std={full_stats['std']:.3g} mm\n"
            f"  FWHM~{full_stats['fwhm_est']:.3g} mm\n"
            f"Other: n={other_stats['n']}\n"
            f"  mean={other_stats['mean']:.3g} mm, std={other_stats['std']:.3g} mm\n"
            f"  FWHM~{other_stats['fwhm_est']:.3g} mm"
        )
        ax.text(
            0.98,
            0.95,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=PAPER_SMALL_TEXT_SIZE,
        )

    axes[0].legend(fontsize=8, loc="best")
    _save_paper_figure(fig, outdir / "pull-xyz-by-gamma-delta-ke.png")

    return metrics


def process_masks(metadata, valid_mask, process_column="process"):
    if metadata is not None and process_column in metadata.columns:
        process = metadata.loc[valid_mask, process_column].astype(str)
        phot_mask = process.str.contains("phot", case=False, na=False).to_numpy()
        compt_mask = process.str.contains("compt", case=False, na=False).to_numpy()
        other_count = int((~(phot_mask | compt_mask)).sum())
    else:
        phot_mask = np.zeros(int(valid_mask.sum()), dtype=bool)
        compt_mask = np.ones(int(valid_mask.sum()), dtype=bool)
        other_count = 0
    return phot_mask, compt_mask, other_count


def _gauss_plus_falling_exp_fixed_x0(x, amp_g, mean_g, sigma_g, amp_e, tau, x0):
    sigma_g = np.maximum(np.abs(sigma_g), LOSS_EPS)
    tau = np.maximum(np.abs(tau), LOSS_EPS)
    return (
        amp_g * np.exp(-0.5 * ((x - mean_g) / sigma_g) ** 2)
        + amp_e * np.exp(-(x - x0) / tau)
    )


def _pe_spectrum_histogram(values, bins=80, mode_min=PE_SPECTRUM_MODE_MIN):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values) & (values > 0)]
    if len(values) < 10:
        return None

    lower = 0.0
    upper = float(np.nanmax(values))
    if not np.isfinite(upper) or upper <= lower:
        return None

    hist_bins = np.linspace(lower, upper, bins + 1)
    counts, edges = np.histogram(values, bins=hist_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    values_for_mode = values[values > mode_min]

    if len(values_for_mode) == 0 or not np.any(counts > 0):
        return {
            "values": values,
            "counts": counts,
            "edges": edges,
            "mode": np.nan,
            "fit_ok": False,
            "reason": "no values above mode threshold",
        }

    mode_counts, _ = np.histogram(values_for_mode, bins=edges)
    return {
        "values": values,
        "counts": counts,
        "edges": edges,
        "mode": float(centers[np.argmax(mode_counts)]),
    }


def _gauss_plus_linear_fixed_x0(x, amp_g, mean_g, sigma_g, intercept, slope, x0):
    sigma_g = np.maximum(np.abs(sigma_g), LOSS_EPS)
    return (
        amp_g * np.exp(-0.5 * ((x - mean_g) / sigma_g) ** 2)
        + intercept
        + slope * (x - x0)
    )


def fit_pe_spectrum_gauss_plus_exp(
    values,
    bins=80,
    mode_min=PE_SPECTRUM_MODE_MIN,
    fit_fraction=PE_SPECTRUM_FIT_FRACTION,
    fit_low_fraction=None,
    fit_high_fraction=None,
    fit_model="gauss-linear",
):
    summary = _pe_spectrum_histogram(values, bins=bins, mode_min=mode_min)
    if summary is None:
        return None

    values = summary["values"]
    counts = summary["counts"]
    edges = summary["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    mode = summary["mode"]
    if not np.isfinite(mode):
        return {
            "values": values,
            "counts": counts,
            "edges": edges,
            "fit_ok": False,
            "reason": summary.get("reason", "invalid mode"),
        }

    min_populated_bins = 6
    low_fraction = float(fit_fraction if fit_low_fraction is None else fit_low_fraction)
    high_fraction = float(fit_fraction if fit_high_fraction is None else fit_high_fraction)
    low_fraction = max(low_fraction, 0.0)
    high_fraction = max(high_fraction, 0.0)
    extra_fraction = 0.0
    fit_mask = np.zeros_like(centers, dtype=bool)
    while extra_fraction <= 2.0:
        fit_lo = max(0.0, (1.0 - low_fraction - extra_fraction) * mode)
        fit_hi = (1.0 + high_fraction + extra_fraction) * mode
        candidate = (centers >= fit_lo) & (centers <= fit_hi) & (counts > 0)
        if np.count_nonzero(candidate) >= min_populated_bins:
            fit_mask = candidate
            break
        extra_fraction += 0.25

    if not np.any(fit_mask):
        fit_mask = (centers >= mode_min) & (counts > 0)

    fit_centers = centers[fit_mask]
    fit_counts = counts[fit_mask]
    if len(fit_centers) < min_populated_bins:
        return {
            "values": values,
            "counts": counts,
            "edges": edges,
            "fit_ok": False,
            "reason": "too few populated fit bins",
            "mode": mode,
        }

    x0 = float(fit_centers[0])
    amp0 = float(np.max(fit_counts))
    mean0 = float(np.average(fit_centers, weights=fit_counts))
    sigma0 = max(0.1 * mode, np.mean(np.diff(edges)), 1.0)

    fit_ok = False
    try:
        from scipy.optimize import curve_fit

        if fit_model == "gauss-exp":
            amp_exp0 = max(0.2 * amp0, 1.0)
            tau0 = max(float(np.mean(fit_centers) - x0), np.mean(np.diff(edges)), 1.0)
            p0 = np.asarray([amp0, mean0, sigma0, amp_exp0, tau0], dtype=float)
            lower_bounds = [0.0, float(fit_centers[0]), LOSS_EPS, 0.0, LOSS_EPS]
            upper_bounds = [np.inf, float(fit_centers[-1]), np.inf, np.inf, np.inf]
            fit_fn = lambda x, amp_g, mean_g, sigma_g, amp_e, tau: _gauss_plus_falling_exp_fixed_x0(
                x, amp_g, mean_g, sigma_g, amp_e, tau, x0
            )
        elif fit_model == "gauss-linear":
            edge_n = max(1, min(3, len(fit_counts) // 4))
            intercept0 = float(np.median(np.r_[fit_counts[:edge_n], fit_counts[-edge_n:]]))
            slope0 = 0.0
            p0 = np.asarray([amp0, mean0, sigma0, intercept0, slope0], dtype=float)
            lower_bounds = [0.0, float(fit_centers[0]), LOSS_EPS, -np.inf, -np.inf]
            upper_bounds = [np.inf, float(fit_centers[-1]), np.inf, np.inf, np.inf]
            fit_fn = lambda x, amp_g, mean_g, sigma_g, intercept, slope: _gauss_plus_linear_fixed_x0(
                x, amp_g, mean_g, sigma_g, intercept, slope, x0
            )
        else:
            raise ValueError(f"Unknown PE spectrum fit model: {fit_model}")

        popt, pcov = curve_fit(
            fit_fn,
            fit_centers,
            fit_counts,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=50000,
        )
        perr = np.sqrt(np.diag(pcov))
        fit_ok = True
        reason = "ok"
    except Exception as exc:
        popt = p0
        perr = np.full(len(p0), np.nan)
        reason = str(exc)

    amp_g, mean_g, sigma_g = popt[:3]
    sigma_g = abs(float(sigma_g))
    fwhm = 2.355 * sigma_g
    resolution_percent = 100.0 * fwhm / mean_g if mean_g > 0 else np.nan

    component_keys = (
        ["amp_g", "mean_g", "sigma_g", "amp_e", "tau"]
        if fit_model == "gauss-exp"
        else ["amp_g", "mean_g", "sigma_g", "linear_intercept", "linear_slope"]
    )

    return {
        "values": values,
        "counts": counts,
        "edges": edges,
        "fit_ok": bool(fit_ok),
        "reason": reason,
        "fit_model": fit_model,
        "mode": float(mode),
        "fit_window": [float(fit_centers[0]), float(fit_centers[-1])],
        "fit_low_fraction": float(low_fraction),
        "fit_high_fraction": float(high_fraction),
        "x0": float(x0),
        "params": np.asarray(popt, dtype=float),
        "param_names": component_keys,
        "errors": np.asarray(perr, dtype=float),
        "gauss_mean": float(mean_g),
        "gauss_sigma": float(sigma_g),
        "gauss_fwhm": float(fwhm),
        "energy_resolution_percent": float(resolution_percent),
    }


def _default_max_pe_feature_labels(
    mode,
    feature_positions=None,
    include_energy_values=False,
):
    """Expected max-pixel spectral features, scaled to the observed photopeak."""
    if not np.isfinite(mode) or mode <= 0:
        return []

    feature_positions = feature_positions or {}

    def _position(name, default):
        value = feature_positions.get(name)
        if value is None:
            return default
        return float(value)

    if include_energy_values:
        labels = {
            "compton_edge": f"Compton Edge\n@ {COMPTON_EDGE_KEV:.0f} keV",
            "klein_nishina_maximum": (
                f"Klein-Nishina\nMaximum @ {KLEIN_NISHINA_MAXIMUM_KEV:.0f} keV"
            ),
            "photopeak": f"Photopeak\n@ {PHOTOPEAK_KEV:.0f} keV",
        }
    else:
        labels = {
            "compton_edge": "Compton Edge",
            "klein_nishina_maximum": "Klein-Nishina\nMaximum",
            "photopeak": "Photopeak",
        }

    return [
        {
            "x": _position("compton_edge", mode * (COMPTON_EDGE_KEV / PHOTOPEAK_KEV)),
            "label": labels["compton_edge"],
        },
        {
            "x": _position(
                "klein_nishina_maximum",
                mode * (KLEIN_NISHINA_MAXIMUM_KEV / PHOTOPEAK_KEV),
            ),
            "label": labels["klein_nishina_maximum"],
        },
        {
            "x": _position("photopeak", mode),
            "label": labels["photopeak"],
        },
    ]


def _default_total_visible_energy_feature_labels():
    return [
        {
            "x": KLEIN_NISHINA_MAXIMUM_KEV,
            "label": (
                f"Klein-Nishina\nMaximum @ "
                f"{KLEIN_NISHINA_MAXIMUM_KEV:.0f} keV"
            ),
            "y_fraction": 0.50,
        },
        {
            "x": PHOTOPEAK_KEV,
            "label": f"Photopeak\n@ {PHOTOPEAK_KEV:.0f} keV",
        },
    ]


def _draw_pe_feature_labels(ax, feature_labels, x_scale=1.0):
    if not feature_labels:
        return []

    x_min, x_max = ax.get_xlim()
    _, y_max = ax.get_ylim()
    text_dx = 0.018 * (x_max - x_min)
    drawn = []
    for feature in feature_labels:
        x = float(feature["x"]) * float(x_scale)
        if not np.isfinite(x) or x < x_min or x > x_max:
            continue
        label = str(feature["label"])
        y_fraction = float(feature.get("y_fraction", 0.96))
        ax.axvline(
            x,
            ymin=0.0,
            ymax=y_fraction,
            color="0.25",
            lw=1.2,
            ls=":",
            alpha=0.85,
        )
        ax.text(
            x - text_dx,
            y_fraction * y_max,
            label,
            rotation=90,
            ha="right",
            va="top",
            fontsize=PAPER_SMALL_TEXT_SIZE,
            color="0.15",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.70),
        )
        drawn.append({"x": x, "label": label.replace("\n", " ")})
    return drawn


def plot_total_pe_energy_spectrum(
    metadata,
    outdir,
    pe_column="total_pe",
    output_filename="total-pe-energy-spectrum.png",
    x_label=r"$N^{\mathrm{total}}_{\mathrm{PE}}$",
    width_text_label=r"$\Delta E/E$",
    show_fit=True,
    annotate_max_pe_features=False,
    max_pe_feature_positions=None,
    max_pe_feature_positions_in_plot_units=False,
    max_pe_feature_energy_values=False,
    spectrum_feature_labels=None,
    x_scale=1.0,
    bins=80,
    mode_min=PE_SPECTRUM_MODE_MIN,
    fit_fraction=PE_SPECTRUM_FIT_FRACTION,
    fit_low_fraction=None,
    fit_high_fraction=None,
    fit_model="gauss-linear",
):
    if metadata is None or pe_column not in metadata.columns:
        print(f"Skipping PE energy spectrum: missing {pe_column}")
        return {}

    if show_fit:
        spectrum = fit_pe_spectrum_gauss_plus_exp(
            metadata[pe_column].to_numpy(dtype=float),
            bins=bins,
            mode_min=mode_min,
            fit_fraction=fit_fraction,
            fit_low_fraction=fit_low_fraction,
            fit_high_fraction=fit_high_fraction,
            fit_model=fit_model,
        )
    else:
        spectrum = _pe_spectrum_histogram(
            metadata[pe_column].to_numpy(dtype=float),
            bins=bins,
            mode_min=mode_min,
        )
        if spectrum is not None:
            spectrum["fit_ok"] = False
            spectrum["reason"] = "fit disabled"
            spectrum["fit_model"] = "none"

    if spectrum is None:
        print("Skipping PE energy spectrum: not enough finite positive values")
        return {}

    values = spectrum["values"]
    counts = spectrum["counts"]
    edges = spectrum["edges"]
    centers = 0.5 * (edges[:-1] + edges[1:])
    x_scale = float(x_scale)
    plot_values = values * x_scale
    plot_edges = edges * x_scale

    fig, ax = plt.subplots(figsize=VALIDATION_SPECTRUM_FIGSIZE)
    ax.hist(
        plot_values,
        bins=plot_edges,
        weights=np.full(len(plot_values), 1.0 / len(plot_values)),
        alpha=0.78,
        color="tab:blue",
        label="simulation" if show_fit else None,
    )

    if show_fit and "params" in spectrum:
        params = spectrum["params"]
        amp_g, mean_g, sigma_g = params[:3]
        norm = 1.0 / len(plot_values)
        amp_g *= norm
        x0 = spectrum["x0"]
        x_fit = np.linspace(spectrum["fit_window"][0], spectrum["fit_window"][1], 500)
        x_fit_plot = x_fit * x_scale
        gauss = amp_g * np.exp(-0.5 * ((x_fit - mean_g) / sigma_g) ** 2)
        if spectrum.get("fit_model") == "gauss-exp":
            amp_e, tau = params[3:]
            amp_e *= norm
            background = amp_e * np.exp(-(x_fit - x0) / max(tau, LOSS_EPS))
            total_label = "Gaussian + exponential fit"
            background_label = "Exponential component"
            background_color = "tab:green"
        else:
            intercept, slope = params[3:]
            intercept *= norm
            slope *= norm
            background = intercept + slope * (x_fit - x0)
            total_label = "Gaussian + linear fit"
            background_label = "Linear background"
            background_color = "tab:green"
        total = gauss + background
        ax.plot(x_fit_plot, total, color="tab:red", lw=2.0, label=total_label)
        ax.plot(x_fit_plot, gauss, color="black", lw=1.4, ls="--", label="Gaussian component")
        ax.plot(x_fit_plot, background, color=background_color, lw=1.4, ls="--", label=background_label)

        txt = rf"{width_text_label} $= {spectrum['energy_resolution_percent']:.2f}\%$"
        ax.text(
            0.03,
            0.72,
            txt,
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=PAPER_SMALL_TEXT_SIZE,
        )

    ax.set_xlabel(x_label, loc="right")
    ax.set_ylabel("Fraction of events / bin", loc="top")
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(bottom=0.0, top=1.15 * ymax if ymax > 0 else None)
    feature_annotations = []
    if annotate_max_pe_features:
        feature_annotations = _draw_pe_feature_labels(
            ax,
            _default_max_pe_feature_labels(
                spectrum.get("mode", np.nan),
                feature_positions=max_pe_feature_positions,
                include_energy_values=max_pe_feature_energy_values,
            ),
            x_scale=1.0 if max_pe_feature_positions_in_plot_units else x_scale,
        )
    if spectrum_feature_labels:
        feature_annotations.extend(
            _draw_pe_feature_labels(ax, spectrum_feature_labels, x_scale=1.0)
        )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=PAPER_SMALL_TEXT_SIZE, loc="upper left")
    _save_paper_figure(fig, outdir / output_filename)

    metrics = {
        "pe_column": pe_column,
        "n": int(len(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "fit_ok": bool(spectrum.get("fit_ok", False)),
        "fit_reason": spectrum.get("reason", ""),
        "fit_model": spectrum.get("fit_model", fit_model),
        "x_scale": float(x_scale),
        "feature_annotations": feature_annotations,
    }
    for key in (
        "mode",
        "fit_window",
        "fit_low_fraction",
        "fit_high_fraction",
        "gauss_mean",
        "gauss_sigma",
        "gauss_fwhm",
        "energy_resolution_percent",
    ):
        if key in spectrum:
            value = spectrum[key]
            metrics[key] = value.tolist() if isinstance(value, np.ndarray) else value
    return metrics


def _infer_true_target_columns_from_columns(columns, output_names):
    """Accept either training CSV coordinates or prediction CSV true columns."""
    columns = set(columns)
    true_cols = [f"{name}_true" for name in output_names]
    if all(col in columns for col in true_cols):
        return true_cols

    gamma_cols = [f"{name}_gamma" for name in output_names]
    if all(col in columns for col in gamma_cols):
        return gamma_cols

    return None


def _infer_true_target_columns(df, output_names):
    return _infer_true_target_columns_from_columns(df.columns, output_names)


def plot_true_z_distribution(
    y_true,
    metadata,
    outdir,
    output_names,
    pixel_geometry=None,
    bins=60,
    min_events=10,
):
    if "z" in output_names:
        z_index = output_names.index("z")
    else:
        z_index = 2

    if metadata is not None and "z_gamma" in metadata.columns:
        z = metadata["z_gamma"].to_numpy(dtype=float)
        z_source = "z_gamma"
    else:
        z = np.asarray(y_true)[:, z_index]
        z_source = f"{output_names[z_index]}_true"

    z = z[np.isfinite(z)]
    if len(z) < min_events:
        print(f"Skipping true-z distribution: only {len(z)} finite events")
        return {"n": int(len(z)), "z_source": z_source}

    if pixel_geometry is not None and "pixel_length_mm" in pixel_geometry:
        half_length = 0.5 * float(pixel_geometry["pixel_length_mm"])
        hist_range = (-half_length, half_length)
    else:
        hist_range = (float(np.nanmin(z)), float(np.nanmax(z)))

    fig, ax = plt.subplots(figsize=VALIDATION_FIGSIZE)
    ax.hist(
        z,
        bins=bins,
        range=hist_range,
        weights=np.full(len(z), 1.0 / len(z)),
        alpha=0.78,
        color="tab:blue",
    )
    ax.set_xlabel("First interaction z (mm)", loc="right")
    ax.set_ylabel("Fraction of events / bin", loc="top")
    _save_paper_figure(fig, outdir / "true-z-distribution.png")

    return {
        "n": int(len(z)),
        "z_source": z_source,
        "mean_mm": float(np.mean(z)),
        "median_mm": float(np.median(z)),
        "std_mm": float(np.std(z, ddof=1)) if len(z) > 1 else np.nan,
        "p05_mm": float(np.percentile(z, 5)),
        "p95_mm": float(np.percentile(z, 95)),
    }


def evaluate_validation_plots(
    y_true,
    outdir,
    output_names,
    metadata=None,
    spatial_indices=(0, 1, 2),
    pixel_geometry=None,
    incident_energy_window_mev=0.001,
    source_radius_mm=13.0,
    source_distance_cm=20.0,
    cone_margin=0.3,
    cone_half_angle_deg=None,
    cone_angle_samples=96,
    pe_spectrum_mode_min=PE_SPECTRUM_MODE_MIN,
    pe_spectrum_fit_fraction=PE_SPECTRUM_FIT_FRACTION,
    pe_spectrum_fit_low_fraction=None,
    pe_spectrum_fit_high_fraction=None,
    pe_spectrum_fit_model="gauss-linear",
    max_pe_feature_positions=None,
):
    """Create validation plots that do not require CNN predictions."""
    outdir = _as_path(outdir)
    output_names = tuple(output_names)
    metrics = {"mode": "validation_only"}

    true_z_metrics = plot_true_z_distribution(
        y_true,
        metadata,
        outdir,
        output_names,
        pixel_geometry=pixel_geometry,
    )
    if true_z_metrics:
        metrics["true_z_distribution"] = true_z_metrics

    pe_spectrum_metrics = plot_total_pe_energy_spectrum(
        metadata,
        outdir,
        width_text_label=r"Gaussian FWHM$/\mu \times 100$",
        mode_min=pe_spectrum_mode_min,
        fit_fraction=pe_spectrum_fit_fraction,
        fit_low_fraction=pe_spectrum_fit_low_fraction,
        fit_high_fraction=pe_spectrum_fit_high_fraction,
        fit_model=pe_spectrum_fit_model,
    )
    if pe_spectrum_metrics:
        metrics["total_pe_energy_spectrum"] = pe_spectrum_metrics
        mode = pe_spectrum_metrics.get("mode", np.nan)
        if np.isfinite(mode) and mode > 0:
            total_visible_energy_metrics = plot_total_pe_energy_spectrum(
                metadata,
                outdir,
                output_filename="total-visible-energy-spectrum.png",
                x_label=r"$E^{\mathrm{total}}_{\mathrm{vis}}$ (keV)",
                width_text_label=r"Gaussian FWHM$/\mu \times 100$",
                mode_min=pe_spectrum_mode_min,
                fit_fraction=pe_spectrum_fit_fraction,
                fit_low_fraction=pe_spectrum_fit_low_fraction,
                fit_high_fraction=pe_spectrum_fit_high_fraction,
                fit_model=pe_spectrum_fit_model,
                x_scale=VISIBLE_ENERGY_CALIBRATION_KEV / mode,
                spectrum_feature_labels=_default_total_visible_energy_feature_labels(),
            )
            if total_visible_energy_metrics:
                metrics["total_visible_energy_spectrum"] = total_visible_energy_metrics

    max_pe_spectrum_metrics = plot_total_pe_energy_spectrum(
        metadata,
        outdir,
        pe_column="max_pe",
        output_filename="max-pixel-pe-spectrum.png",
        x_label=r"$N^{\mathrm{max}}_{\mathrm{PE}}$",
        width_text_label=r"$\mathrm{FWHM}/\mu$",
        show_fit=False,
        annotate_max_pe_features=True,
        max_pe_feature_positions=max_pe_feature_positions,
        mode_min=pe_spectrum_mode_min,
        fit_fraction=pe_spectrum_fit_fraction,
        fit_low_fraction=pe_spectrum_fit_low_fraction,
        fit_high_fraction=pe_spectrum_fit_high_fraction,
        fit_model=pe_spectrum_fit_model,
    )
    if max_pe_spectrum_metrics:
        metrics["max_pixel_pe_spectrum"] = max_pe_spectrum_metrics

    gamma_depth_metrics = plot_gamma_first_interaction_depth_validation(
        y_true,
        metadata,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        pixel_geometry=pixel_geometry,
        incident_energy_window_mev=incident_energy_window_mev,
        source_radius_mm=source_radius_mm,
        source_distance_cm=source_distance_cm,
        cone_margin=cone_margin,
        cone_half_angle_deg=cone_half_angle_deg,
        cone_angle_samples=cone_angle_samples,
    )
    if gamma_depth_metrics:
        metrics["gamma_first_interaction_depth_validation"] = gamma_depth_metrics

    incident_angle_metrics = plot_gamma_incident_angle_distribution(metadata, outdir)
    if incident_angle_metrics:
        metrics["gamma_incident_angle_distribution"] = incident_angle_metrics

    report_path = write_html_report(outdir, metrics)
    metrics["html_report"] = str(report_path)
    return metrics


def assign_true_pixel(metadata, pixel_geometry):
    if metadata is None or "x_gamma" not in metadata.columns or "y_gamma" not in metadata.columns:
        raise RuntimeError("Per-pixel diagnostics require x_gamma and y_gamma columns")

    image_size = pixel_geometry["image_size"]
    pitch = pixel_geometry["pitch_mm"]
    offset = pixel_geometry["offset_mm"]
    pixel_size = pixel_geometry["pixel_size_mm"]

    x = metadata["x_gamma"].to_numpy(dtype=float)
    y = metadata["y_gamma"].to_numpy(dtype=float)

    ix = np.rint((x + offset) / pitch).astype(int)
    iy = np.rint((y + offset) / pitch).astype(int)
    cx = ix * pitch - offset
    cy = iy * pitch - offset

    inside_pixel = (
        (ix >= 0) & (ix < image_size) &
        (iy >= 0) & (iy < image_size) &
        (np.abs(x - cx) <= 0.5 * pixel_size) &
        (np.abs(y - cy) <= 0.5 * pixel_size)
    )
    return ix, iy, inside_pixel


def assign_pixel_region_categories(metadata, pixel_geometry):
    ix, iy, inside_pixel = assign_true_pixel(metadata, pixel_geometry)
    image_size = pixel_geometry["image_size"]

    if image_size % 2 == 0:
        centre_indices = {image_size // 2 - 1, image_size // 2}
    else:
        centre_indices = {image_size // 2}

    ix_central = np.isin(ix, list(centre_indices))
    iy_central = np.isin(iy, list(centre_indices))
    central = inside_pixel & ix_central & iy_central

    on_x_edge = (ix == 0) | (ix == image_size - 1)
    on_y_edge = (iy == 0) | (iy == image_size - 1)
    corner = inside_pixel & on_x_edge & on_y_edge
    edge_non_corner = inside_pixel & (on_x_edge | on_y_edge) & ~corner
    inner_non_central = inside_pixel & ~central & ~edge_non_corner & ~corner

    return [
        ("central_four", "Central four pixels", central),
        ("inner_non_edge", "Interior non-edge pixels", inner_non_central),
        ("edge_non_corner", "Edge pixels, not corners", edge_non_corner),
        ("corner", "Corner pixels", corner),
    ]


def _finite_percentile_range(values, lower=0.5, upper=99.5, pad_fraction=0.05):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return -0.5, 0.5

    lo, hi = np.nanpercentile(values, [lower, upper])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = float(np.nanmin(values)), float(np.nanmax(values))
    if lo == hi:
        lo -= 0.5
        hi += 0.5

    pad = pad_fraction * (hi - lo)
    return lo - pad, hi + pad


def plot_residuals_by_pixel_region(
    y_true,
    y_pred,
    metadata,
    outdir,
    output_names,
    pixel_geometry,
    spatial_indices=(0, 1, 2),
    min_events=30,
):
    if metadata is None or pixel_geometry is None:
        return {}

    try:
        categories = assign_pixel_region_categories(metadata, pixel_geometry)
    except RuntimeError as exc:
        print(f"Skipping pixel-region residual diagnostics: {exc}")
        return {}

    residual = np.asarray(y_true) - np.asarray(y_pred)
    spatial_indices = tuple(spatial_indices)
    colors = ["tab:blue", "tab:orange", "tab:green"]
    fig, axes = plt.subplots(
        len(categories),
        len(spatial_indices),
        figsize=(5.0 * len(spatial_indices), 3.6 * len(categories)),
        squeeze=False,
    )
    metrics = {}

    axis_bins = {}
    for idx in spatial_indices:
        values = residual[:, idx]
        lo, hi = _finite_percentile_range(values)
        axis_bins[idx] = np.linspace(lo, hi, 50)

    for row, (key, label, region_mask) in enumerate(categories):
        metrics[key] = {"label": label, "n_events": int(np.sum(region_mask)), "axes": {}}

        for col, (idx, color) in enumerate(zip(spatial_indices, colors)):
            ax = axes[row, col]
            name = output_names[idx]
            values = residual[region_mask, idx]
            values = values[np.isfinite(values)]
            n = int(len(values))
            bins = axis_bins[idx]

            if n == 0:
                ax.text(0.5, 0.5, "No events", transform=ax.transAxes, ha="center", va="center")
                ax.set_axis_off()
                continue

            ax.hist(values, bins=bins, density=True, alpha=0.48, color=color, label="residuals")
            fit = _fit_gaussian_pdf(values, bins)
            if n >= min_events and np.isfinite(fit["sigma"]) and fit["sigma"] > 0:
                x_fit = np.linspace(bins[0], bins[-1], 300)
                ax.plot(
                    x_fit,
                    _gaussian(x_fit, fit["amplitude"], fit["mean"], fit["sigma"]),
                    color=color,
                    alpha=0.95,
                    lw=2.0,
                    label="Gaussian fit",
                )
            ax.axvline(0.0, color="black", ls="--", lw=1)
            ax.set_xlabel(f"{name}_true - {name}_pred (mm)", loc="right")
            ax.set_ylabel("Density", loc="top")
            ax.text(
                0.98,
                0.95,
                (
                    f"{label}\n"
                    f"n = {n}\n"
                    rf"$\mu = {fit['mean']:.3g}\,\mathrm{{mm}}$"
                    "\n"
                    rf"$\mathrm{{FWHM}} = {fit['fwhm']:.3g}\,\mathrm{{mm}}$"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
                fontsize=8,
            )
            ax.legend(fontsize=8, loc="best")

            metrics[key]["axes"][name] = {
                "n": n,
                "fit_ok": bool(fit["fit_ok"]),
                "mean_mm": float(fit["mean"]),
                "sigma_mm": float(fit["sigma"]),
                "fwhm_mm": float(fit["fwhm"]),
            }

    _save_paper_figure(fig, outdir / "residuals-by-pixel-region.png")
    return metrics


def plot_residual_trends_vs_true_z_by_pixel_region(
    y_true,
    y_pred,
    metadata,
    outdir,
    output_names,
    pixel_geometry,
    spatial_indices=(0, 1, 2),
    n_z_bins=8,
    min_events_per_bin=30,
):
    if metadata is None or pixel_geometry is None:
        return {}

    try:
        categories = assign_pixel_region_categories(metadata, pixel_geometry)
    except RuntimeError as exc:
        print(f"Skipping pixel-region residual trends: {exc}")
        return {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    spatial_indices = tuple(spatial_indices)
    z_index = spatial_indices[2]
    true_z = y_true[:, z_index]
    finite_z = np.isfinite(true_z)
    if finite_z.sum() < min_events_per_bin:
        return {}

    z_min = float(np.nanmin(true_z[finite_z]))
    z_max = float(np.nanmax(true_z[finite_z]))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
        return {}

    edges = np.linspace(z_min, z_max, n_z_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    colors = {
        "central_four": "tab:blue",
        "inner_non_edge": "tab:orange",
        "edge_non_corner": "tab:green",
        "corner": "tab:red",
    }

    fig_fwhm, axes_fwhm = plt.subplots(
        1,
        len(spatial_indices),
        figsize=(5.0 * len(spatial_indices), 4.2),
        squeeze=False,
    )
    fig_mean, axes_mean = plt.subplots(
        1,
        len(spatial_indices),
        figsize=(5.0 * len(spatial_indices), 4.2),
        squeeze=False,
    )
    axes_fwhm = axes_fwhm.reshape(-1)
    axes_mean = axes_mean.reshape(-1)
    metrics = {
        "z_bin_edges": edges.tolist(),
        "z_bin_centers": centers.tolist(),
        "min_events_per_bin": int(min_events_per_bin),
        "categories": {},
    }

    for key, label, region_mask in categories:
        metrics["categories"][key] = {"label": label, "axes": {}}

        for col, idx in enumerate(spatial_indices):
            name = output_names[idx]
            fwhm_values = []
            mean_values = []
            n_values = []
            fit_ok_values = []

            for lo, hi in zip(edges[:-1], edges[1:]):
                in_bin = region_mask & finite_z & (true_z >= lo) & (true_z < hi)
                if hi == edges[-1]:
                    in_bin = region_mask & finite_z & (true_z >= lo) & (true_z <= hi)

                values = residual[in_bin, idx]
                values = values[np.isfinite(values)]
                n = int(len(values))
                n_values.append(n)

                if n < min_events_per_bin:
                    fwhm_values.append(np.nan)
                    mean_values.append(np.nan)
                    fit_ok_values.append(False)
                    continue

                fit = _fit_gaussian_to_histogram(values, bins=40)
                fwhm_values.append(float(fit["fwhm"]))
                mean_values.append(float(fit["mean"]))
                fit_ok_values.append(bool(fit["fit_ok"]))

            axes_fwhm[col].plot(
                centers,
                fwhm_values,
                marker="o",
                ms=4,
                lw=1.8,
                color=colors.get(key),
                label=label,
            )
            axes_mean[col].plot(
                centers,
                mean_values,
                marker="o",
                ms=4,
                lw=1.8,
                color=colors.get(key),
                label=label,
            )

            metrics["categories"][key]["axes"][name] = {
                "fwhm_mm": [float(v) if np.isfinite(v) else np.nan for v in fwhm_values],
                "mean_mm": [float(v) if np.isfinite(v) else np.nan for v in mean_values],
                "n_events": n_values,
                "fit_ok": fit_ok_values,
            }

    for ax, idx in zip(axes_fwhm, spatial_indices):
        ax.set_xlabel("True z position (mm)", loc="right")
        ax.set_ylabel(f"{output_names[idx]} residual FWHM (mm)", loc="top")
        ax.legend(fontsize=8, loc="best")

    for ax, idx in zip(axes_mean, spatial_indices):
        ax.axhline(0.0, color="black", ls="--", lw=1)
        ax.set_xlabel("True z position (mm)", loc="right")
        ax.set_ylabel(f"{output_names[idx]} residual mean (mm)", loc="top")
        ax.legend(fontsize=8, loc="best")

    _save_paper_figure(fig_fwhm, outdir / "residual-fwhm-vs-true-z-by-pixel-region.png")
    _save_paper_figure(fig_mean, outdir / "residual-mean-vs-true-z-by-pixel-region.png")
    return metrics


def _residual_trend_in_bins(
    x_values,
    residual_values,
    bin_edges,
    min_events_per_bin,
):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    fwhm_values = []
    mean_values = []
    n_values = []
    fit_ok_values = []

    finite = np.isfinite(x_values) & np.isfinite(residual_values)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = finite & (x_values >= lo) & (x_values < hi)
        if hi == bin_edges[-1]:
            in_bin = finite & (x_values >= lo) & (x_values <= hi)

        values = residual_values[in_bin]
        values = values[np.isfinite(values)]
        n = int(len(values))
        n_values.append(n)

        if n < min_events_per_bin:
            fwhm_values.append(np.nan)
            mean_values.append(np.nan)
            fit_ok_values.append(False)
            continue

        fit = _fit_gaussian_to_histogram(values, bins=40)
        fwhm_values.append(float(fit["fwhm"]))
        mean_values.append(float(fit["mean"]))
        fit_ok_values.append(bool(fit["fit_ok"]))

    return {
        "centers": centers,
        "fwhm": np.asarray(fwhm_values, dtype=float),
        "mean": np.asarray(mean_values, dtype=float),
        "n": n_values,
        "fit_ok": fit_ok_values,
    }


def _make_equal_width_edges(values, n_bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return None
    return np.linspace(lo, hi, n_bins + 1)


def _make_quantile_edges(values, n_bins):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return None
    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if len(edges) < 3:
        return _make_equal_width_edges(values, n_bins)
    return edges


def _plot_residual_metric_trends(
    trend_results,
    output_names,
    spatial_indices,
    x_label,
    outdir,
    fwhm_filename,
    mean_filename,
):
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
    fig_fwhm, axes_fwhm = plt.subplots(
        1,
        len(spatial_indices),
        figsize=(5.0 * len(spatial_indices), 4.2),
        squeeze=False,
    )
    fig_mean, axes_mean = plt.subplots(
        1,
        len(spatial_indices),
        figsize=(5.0 * len(spatial_indices), 4.2),
        squeeze=False,
    )
    axes_fwhm = axes_fwhm.reshape(-1)
    axes_mean = axes_mean.reshape(-1)

    for axis_pos, idx in enumerate(spatial_indices):
        name = output_names[idx]
        ax_fwhm = axes_fwhm[axis_pos]
        ax_mean = axes_mean[axis_pos]

        for category_index, (category_key, category) in enumerate(trend_results.items()):
            axis_result = category["axes"].get(name)
            if axis_result is None:
                continue

            color = colors[category_index % len(colors)]
            label = category["label"]
            centers = np.asarray(axis_result["centers"], dtype=float)
            fwhm = np.asarray(axis_result["fwhm_mm"], dtype=float)
            mean = np.asarray(axis_result["mean_mm"], dtype=float)

            ax_fwhm.plot(
                centers,
                fwhm,
                marker="o",
                ms=4,
                lw=1.8,
                color=color,
                label=label,
            )
            ax_mean.plot(
                centers,
                mean,
                marker="o",
                ms=4,
                lw=1.8,
                color=color,
                label=label,
            )

        ax_fwhm.set_xlabel(x_label, loc="right")
        ax_fwhm.set_ylabel(f"{name} residual FWHM (mm)", loc="top")
        ax_fwhm.legend(fontsize=8, loc="best")

        ax_mean.axhline(0.0, color="black", ls="--", lw=1)
        ax_mean.set_xlabel(x_label, loc="right")
        ax_mean.set_ylabel(f"{name} residual mean (mm)", loc="top")
        ax_mean.legend(fontsize=8, loc="best")

    _save_paper_figure(fig_fwhm, outdir / fwhm_filename)
    _save_paper_figure(fig_mean, outdir / mean_filename)


def plot_residual_trends_vs_energy_proxy(
    y_true,
    y_pred,
    metadata,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    n_bins=8,
    min_events_per_bin=30,
):
    """Plot Gaussian-fit residual FWHM/mean versus measured or truth energy proxy."""
    if metadata is None:
        return {}

    energy_candidates = [
        ("total_pe", "Total PE"),
        ("gamma_delta_ke", "First-interaction gamma energy loss (MeV)"),
        ("gamma_edep", "First-interaction deposited energy (MeV)"),
        ("gamma_incident_energy", "Incident gamma energy (MeV)"),
    ]
    energy_column = None
    energy_label = None
    for column, label in energy_candidates:
        if column in metadata.columns:
            energy_column = column
            energy_label = label
            break

    if energy_column is None:
        print("Skipping residual trends vs energy: no energy proxy column found")
        return {}

    x_values = metadata[energy_column].to_numpy(dtype=float)
    finite_x = np.isfinite(x_values)
    if finite_x.sum() < min_events_per_bin:
        print("Skipping residual trends vs energy: too few finite energy values")
        return {}

    edges = _make_quantile_edges(x_values[finite_x], n_bins)
    if edges is None:
        print("Skipping residual trends vs energy: invalid energy range")
        return {}

    residual = np.asarray(y_true) - np.asarray(y_pred)
    metrics = {
        "energy_column": energy_column,
        "energy_label": energy_label,
        "bin_edges": edges.tolist(),
        "min_events_per_bin": int(min_events_per_bin),
        "categories": {
            "all": {
                "label": "All events",
                "axes": {},
            }
        },
    }

    for idx in spatial_indices:
        name = output_names[idx]
        trend = _residual_trend_in_bins(
            x_values,
            residual[:, idx],
            edges,
            min_events_per_bin,
        )
        metrics["categories"]["all"]["axes"][name] = {
            "centers": trend["centers"].tolist(),
            "fwhm_mm": [float(v) if np.isfinite(v) else np.nan for v in trend["fwhm"]],
            "mean_mm": [float(v) if np.isfinite(v) else np.nan for v in trend["mean"]],
            "n_events": trend["n"],
            "fit_ok": trend["fit_ok"],
        }

    _plot_residual_metric_trends(
        metrics["categories"],
        output_names,
        spatial_indices,
        energy_label,
        outdir,
        "residual-fwhm-vs-energy-proxy.png",
        "residual-mean-vs-energy-proxy.png",
    )
    return metrics


def _process_category_masks(metadata, process_column, base_valid):
    if metadata is None or process_column not in metadata.columns:
        return []

    process = metadata[process_column].astype(str)
    phot = process.str.contains("phot", case=False, na=False).to_numpy() & base_valid
    compt = process.str.contains("compt", case=False, na=False).to_numpy() & base_valid
    categories = []
    if np.any(phot):
        categories.append(("photoelectric", "Photoelectric first interaction", phot))
    if np.any(compt):
        categories.append(("compton", "Compton first interaction", compt))
    return categories


def plot_residual_trends_vs_true_z_by_process(
    y_true,
    y_pred,
    metadata,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    process_column="process",
    n_z_bins=8,
    min_events_per_bin=30,
):
    """Plot Gaussian-fit residual FWHM/mean versus true z split by PE/Compton."""
    if metadata is None:
        return {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    z_index = spatial_indices[2]
    true_z = y_true[:, z_index]
    finite_z = np.isfinite(true_z)
    categories = _process_category_masks(metadata, process_column, finite_z)
    if not categories:
        print("Skipping residual trends vs true z by process: no PE/Compton labels found")
        return {}

    edges = _make_equal_width_edges(true_z[finite_z], n_z_bins)
    if edges is None:
        print("Skipping residual trends vs true z by process: invalid z range")
        return {}

    metrics = {
        "process_column": process_column,
        "z_bin_edges": edges.tolist(),
        "min_events_per_bin": int(min_events_per_bin),
        "categories": {},
    }

    for key, label, mask in categories:
        metrics["categories"][key] = {
            "label": label,
            "n_events": int(mask.sum()),
            "axes": {},
        }
        for idx in spatial_indices:
            name = output_names[idx]
            trend = _residual_trend_in_bins(
                true_z[mask],
                residual[mask, idx],
                edges,
                min_events_per_bin,
            )
            metrics["categories"][key]["axes"][name] = {
                "centers": trend["centers"].tolist(),
                "fwhm_mm": [float(v) if np.isfinite(v) else np.nan for v in trend["fwhm"]],
                "mean_mm": [float(v) if np.isfinite(v) else np.nan for v in trend["mean"]],
                "n_events": trend["n"],
                "fit_ok": trend["fit_ok"],
            }

    _plot_residual_metric_trends(
        metrics["categories"],
        output_names,
        spatial_indices,
        "True z position (mm)",
        outdir,
        "residual-fwhm-vs-true-z-by-process.png",
        "residual-mean-vs-true-z-by-process.png",
    )
    return metrics


def plot_resolution_map(metrics, pixel_geometry, path):
    image_size = pixel_geometry["image_size"]
    fwhm_map = np.full((image_size, image_size), np.nan)
    n_map = np.zeros((image_size, image_size), dtype=int)

    for px in range(image_size):
        for py in range(image_size):
            key = f"pixel_{px}_{py}"
            pixel_metrics = metrics.get(key, {})
            fwhm_map[py, px] = pixel_metrics.get("fit_fwhm", np.nan)
            n_map[py, px] = int(pixel_metrics.get("n_test", 0))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(fwhm_map, origin="lower", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("DOI FWHM resolution (mm)")
    ax.set_xticks(np.arange(image_size))
    ax.set_yticks(np.arange(image_size))
    ax.set_xlabel("LYSO pixel x index", loc="right")
    ax.set_ylabel("LYSO pixel y index", loc="top")

    for py in range(image_size):
        for px in range(image_size):
            value = fwhm_map[py, px]
            text = f"{value:.2f}\n(n={n_map[py, px]})" if np.isfinite(value) else f"n={n_map[py, px]}"
            ax.text(px, py, text, ha="center", va="center", color="white", fontsize=7)

    _save_paper_figure(fig, path)


def plot_per_pixel_doi_diagnostics(y_true, y_pred, metadata, outdir, z_index, pixel_geometry, min_events=5):
    pixel_dir = outdir / "per-pixel"
    pixel_dir.mkdir(parents=True, exist_ok=True)

    try:
        ix, iy, inside_pixel = assign_true_pixel(metadata, pixel_geometry)
    except RuntimeError as exc:
        print(f"Skipping per-pixel diagnostics: {exc}")
        return {}

    image_size = pixel_geometry["image_size"]
    z_true = y_true[:, z_index]
    z_pred = y_pred[:, z_index]
    metrics = {}
    skipped = 0

    for px in range(image_size):
        for py in range(image_size):
            mask = (ix == px) & (iy == py) & inside_pixel
            n = int(mask.sum())
            key = f"pixel_{px}_{py}"
            metrics[key] = {"n_test": n}
            if n < min_events:
                skipped += 1
                continue

            residual = z_true[mask] - z_pred[mask]
            fit_metrics = plot_residual_histogram(
                residual,
                pixel_dir / f"pull_z_pixel_{px}_{py}.png",
                f"z residuals: pixel ({px}, {py}), n={n}",
                "z_true - z_pred (mm)",
            )
            metrics[key].update(fit_metrics)

    plot_resolution_map(metrics, pixel_geometry, pixel_dir / "doi-resolution-fwhm-map.png")
    if skipped:
        print(f"Skipped {skipped} per-pixel z pull plots with fewer than {min_events} test events")
    return metrics


def plot_uncertainty_resolution_sensitivity_tradeoff(
    y_true,
    y_pred,
    predicted_uncertainties,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    min_selected_events=20,
):
    """Plot resolution/sensitivity trade-off from a scalar uncertainty quality variable.

    The event quality is

        q = sqrt(sum_i (sigma_i / median(sigma_i))^2)

    using only the requested spatial axes. Events are selected with q < q_cut.
    """
    if predicted_uncertainties is None:
        return {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    predicted_uncertainties = np.asarray(predicted_uncertainties)
    spatial_indices = tuple(spatial_indices)

    if predicted_uncertainties.shape[0] != y_pred.shape[0]:
        print("Skipping uncertainty trade-off plot: uncertainty array length does not match predictions")
        return {}
    if predicted_uncertainties.shape[1] <= max(spatial_indices):
        print("Skipping uncertainty trade-off plot: uncertainty array has too few columns")
        return {}

    residual = y_true[:, list(spatial_indices)] - y_pred[:, list(spatial_indices)]
    sigma = predicted_uncertainties[:, list(spatial_indices)]
    valid = np.all(np.isfinite(residual), axis=1) & np.all(np.isfinite(sigma), axis=1) & np.all(sigma > 0, axis=1)
    if valid.sum() < min_selected_events:
        print("Skipping uncertainty trade-off plot: too few finite uncertainty predictions")
        return {}

    sigma_valid = sigma[valid]
    residual_valid = residual[valid]
    scales = np.median(sigma_valid, axis=0)
    scales = np.maximum(scales, LOSS_EPS)
    q = np.sqrt(np.sum((sigma_valid / scales) ** 2, axis=1))

    thresholds = np.unique(np.quantile(q, np.linspace(0.05, 0.95, 19)))
    if len(thresholds) < 2:
        print("Skipping uncertainty trade-off plot: q has insufficient dynamic range")
        return {}

    selected_fractions = []
    fwhm_by_axis = {output_names[idx]: [] for idx in spatial_indices}
    n_selected = []

    for threshold in thresholds:
        selected = q <= threshold
        n = int(selected.sum())
        n_selected.append(n)
        selected_fractions.append(float(n / len(q)))

        for axis_pos, idx in enumerate(spatial_indices):
            name = output_names[idx]
            if n < min_selected_events:
                fwhm = np.nan
            else:
                axis_residual = residual_valid[selected, axis_pos]
                std = np.std(axis_residual, ddof=1) if len(axis_residual) > 1 else np.nan
                fwhm = 2.355 * std
            fwhm_by_axis[name].append(float(fwhm))

    fig, ax_res = plt.subplots(figsize=(7.2, 4.8))
    colors = ["tab:blue", "tab:orange", "tab:green"]
    for color, idx in zip(colors, spatial_indices):
        name = output_names[idx]
        ax_res.plot(
            thresholds,
            fwhm_by_axis[name],
            marker="o",
            ms=4,
            lw=1.6,
            color=color,
            label=f"{name} residual FWHM",
        )

    ax_sens = ax_res.twinx()
    ax_sens.plot(
        thresholds,
        selected_fractions,
        marker="s",
        ms=4,
        lw=1.6,
        color="black",
        ls="--",
        label="selected fraction",
    )

    ax_res.set_xlabel("Event-quality cut $q < q_{\\mathrm{cut}}$", loc="right")
    ax_res.set_ylabel("Residual FWHM (mm)", loc="top")
    ax_sens.set_ylabel("Selected test-event fraction", loc="top")
    ax_sens.set_ylim(0.0, 1.02)
    lines_res, labels_res = ax_res.get_legend_handles_labels()
    lines_sens, labels_sens = ax_sens.get_legend_handles_labels()
    ax_res.legend(lines_res + lines_sens, labels_res + labels_sens, fontsize=8, loc="best")

    _save_paper_figure(fig, outdir / "uncertainty-resolution-sensitivity-tradeoff.png")

    return {
        "quality_variable": "sqrt(sum_i (sigma_i / median_sigma_i)^2)",
        "spatial_axes": [output_names[idx] for idx in spatial_indices],
        "median_sigma_scales_mm": {
            output_names[idx]: float(scales[pos]) for pos, idx in enumerate(spatial_indices)
        },
        "q_thresholds": thresholds.tolist(),
        "selected_fraction": selected_fractions,
        "n_selected": n_selected,
        "fwhm_mm": fwhm_by_axis,
    }


def plot_uncertainty_diagnostics(
    y_true,
    y_pred,
    predicted_uncertainties,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
):
    if predicted_uncertainties is None:
        return {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    predicted_uncertainties = np.asarray(predicted_uncertainties)
    output_names = tuple(output_names)
    spatial_indices = tuple(spatial_indices)

    if predicted_uncertainties.shape[0] != y_pred.shape[0]:
        print("Skipping uncertainty diagnostics: uncertainty array length does not match predictions")
        return {}
    if predicted_uncertainties.shape[1] <= max(spatial_indices):
        print("Skipping uncertainty diagnostics: uncertainty array has too few columns")
        return {}

    residual = y_true - y_pred
    safe_sigma = np.maximum(predicted_uncertainties, LOSS_EPS)
    normalized_pull = residual / safe_sigma
    metrics = {}

    fig, axes = plt.subplots(1, len(spatial_indices), figsize=(5 * len(spatial_indices), 4))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, spatial_indices):
        name = output_names[idx]
        values = normalized_pull[:, idx]
        values = values[np.isfinite(values)]
        mean = float(np.mean(values)) if len(values) else np.nan
        std = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan

        ax.hist(values, bins=50, alpha=0.85)
        ax.axvline(0.0, color="black", ls="--", lw=1)
        ax.set_xlabel(f"({name}_true - {name}_pred) / sigma_{name}", loc="right")
        ax.set_ylabel("Counts", loc="top")
        ax.text(
            0.98,
            0.95,
            f"mean = {mean:.3g}\nstd = {std:.3g}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=9,
        )

        metrics[name] = {
            "mean_sigma_mm": float(np.mean(safe_sigma[:, idx])),
            "median_sigma_mm": float(np.median(safe_sigma[:, idx])),
            "pull_mean": mean,
            "pull_std": std,
        }

    _save_paper_figure(fig, outdir / "uncertainty-normalized-pulls.png")

    fig, axes = plt.subplots(1, len(spatial_indices), figsize=(5 * len(spatial_indices), 4))
    axes = np.asarray(axes).reshape(-1)
    for ax, idx in zip(axes, spatial_indices):
        name = output_names[idx]
        ax.hist(safe_sigma[:, idx], bins=50, alpha=0.85)
        ax.set_xlabel(f"Predicted sigma_{name} (mm)", loc="right")
        ax.set_ylabel("Counts", loc="top")

    _save_paper_figure(fig, outdir / "predicted-uncertainties.png")

    fig, axes = plt.subplots(1, len(spatial_indices), figsize=(5 * len(spatial_indices), 4))
    axes = np.asarray(axes).reshape(-1)
    rng = np.random.default_rng(12345)
    for ax, idx in zip(axes, spatial_indices):
        name = output_names[idx]
        abs_residual = np.abs(residual[:, idx])
        sigma = safe_sigma[:, idx]
        finite = np.isfinite(abs_residual) & np.isfinite(sigma)
        if finite.sum() <= 2:
            continue

        corr = float(np.corrcoef(abs_residual[finite], sigma[finite])[0, 1])
        rank_corr = np.nan
        try:
            from scipy.stats import spearmanr

            rank_corr = float(spearmanr(sigma[finite], abs_residual[finite]).correlation)
        except Exception:
            pass

        x_limit = max(float(np.nanpercentile(sigma[finite], 99.5)), LOSS_EPS)
        y_limit = max(float(np.nanpercentile(abs_residual[finite], 99.5)), LOSS_EPS)
        visible = (
            finite
            & (sigma >= 0.0)
            & (sigma <= x_limit)
            & (abs_residual >= 0.0)
            & (abs_residual <= y_limit)
        )

        visible_indices = np.flatnonzero(visible)
        max_scatter = 70000
        if len(visible_indices) > max_scatter:
            draw = rng.choice(visible_indices, size=max_scatter, replace=False)
            scatter_label = f"events, random {max_scatter:,}/{len(visible_indices):,}"
            scatter_alpha = 0.035
        else:
            draw = visible_indices
            scatter_label = "events"
            scatter_alpha = 0.10 if len(draw) < 10000 else 0.045

        ax.scatter(
            sigma[draw],
            abs_residual[draw],
            s=5,
            alpha=scatter_alpha,
            color="tab:blue",
            edgecolors="none",
            rasterized=True,
            label=scatter_label,
        )
        ax.plot(
            [0.0, min(x_limit, y_limit)],
            [0.0, min(x_limit, y_limit)],
            color="black",
            ls="--",
            lw=1,
            label="|residual| = sigma",
        )

        bin_edges = np.quantile(sigma[visible], np.linspace(0.0, 1.0, 13))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) > 2:
            centers = []
            medians = []
            p16 = []
            p84 = []
            p05 = []
            p95 = []
            for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
                mask = visible & (sigma >= lo) & (sigma <= hi)
                if mask.sum() < 30:
                    continue
                centers.append(float(np.median(sigma[mask])))
                medians.append(float(np.median(abs_residual[mask])))
                p16.append(float(np.percentile(abs_residual[mask], 16)))
                p84.append(float(np.percentile(abs_residual[mask], 84)))
                p05.append(float(np.percentile(abs_residual[mask], 5)))
                p95.append(float(np.percentile(abs_residual[mask], 95)))
            if centers:
                centers = np.asarray(centers)
                medians = np.asarray(medians)
                p16 = np.asarray(p16)
                p84 = np.asarray(p84)
                p05 = np.asarray(p05)
                p95 = np.asarray(p95)
                ax.fill_between(
                    centers,
                    p05,
                    p95,
                    color="tab:red",
                    alpha=0.10,
                    label="binned 5-95%",
                )
                ax.fill_between(
                    centers,
                    p16,
                    p84,
                    color="tab:red",
                    alpha=0.20,
                    label="binned 16-84%",
                )
                ax.errorbar(
                    centers,
                    medians,
                    fmt="o-",
                    color="tab:red",
                    ms=4,
                    lw=1.5,
                    label="binned median",
                )

        ax.set_xlim(0.0, x_limit)
        ax.set_ylim(0.0, y_limit)
        ax.set_xlabel(f"Predicted sigma_{name} (mm)", loc="right")
        ax.set_ylabel(f"|{name}_true - {name}_pred| (mm)", loc="top")
        ax.legend(fontsize=8, loc="best")
        metrics[name]["abs_residual_sigma_corr"] = corr
        metrics[name]["abs_residual_sigma_spearman"] = rank_corr
        metrics[name]["abs_residual_sigma_axis_percentile"] = 99.5
        metrics[name]["abs_residual_sigma_plot_style"] = "subsampled_scatter_with_binned_quantiles"
        metrics[name]["abs_residual_sigma_scatter_points"] = int(len(draw))

    _save_paper_figure(fig, outdir / "abs-residual-vs-predicted-uncertainty.png")

    return metrics


def plot_residuals_with_uncertainty_cut(
    y_true,
    y_pred,
    predicted_uncertainties,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
):
    if predicted_uncertainties is None:
        return {}

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    predicted_uncertainties = np.asarray(predicted_uncertainties)
    output_names = tuple(output_names)
    spatial_indices = tuple(spatial_indices)

    residual = y_true - y_pred
    safe_sigma = np.maximum(predicted_uncertainties, LOSS_EPS)

    metrics = {}
    fig, axes = plt.subplots(1, len(spatial_indices), figsize=(5.3 * len(spatial_indices), 4.2))
    axes = np.asarray(axes).reshape(-1)

    for ax, idx in zip(axes, spatial_indices):
        name = output_names[idx]
        all_residual = residual[:, idx]
        if safe_sigma.shape[1] <= idx:
            continue
        sigma_finite = safe_sigma[np.isfinite(safe_sigma[:, idx]), idx]
        uncertainty_cut_mm = float(np.median(sigma_finite)) if len(sigma_finite) > 0 else np.nan
        cut_mask = (
            np.isfinite(all_residual)
            & np.isfinite(safe_sigma[:, idx])
            & (safe_sigma[:, idx] < uncertainty_cut_mm)
        )
        cut_residual = all_residual[cut_mask]

        finite_all = all_residual[np.isfinite(all_residual)]
        if len(finite_all) == 0:
            continue

        lo, hi = np.nanpercentile(finite_all, [0.5, 99.5])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo, hi = float(np.nanmin(finite_all)), float(np.nanmax(finite_all))
        if lo == hi:
            lo -= 0.5
            hi += 0.5

        bins = np.linspace(lo, hi, 50)
        ax.hist(finite_all, bins=bins, density=True, alpha=0.45, color="tab:gray", label="Test set")
        ax.hist(
            cut_residual,
            bins=bins,
            density=True,
            alpha=0.55,
            color="tab:blue",
            label=rf"Test set with $\sigma_{{{name}}} < \tilde{{\sigma}}_{{{name}}}$",
        )
        all_fit = _fit_gaussian_pdf(finite_all, bins)
        cut_fit = _fit_gaussian_pdf(cut_residual, bins)
        x_fit = np.linspace(lo, hi, 400)
        if np.isfinite(all_fit["sigma"]) and all_fit["sigma"] > 0:
            ax.plot(
                x_fit,
                _gaussian(x_fit, all_fit["amplitude"], all_fit["mean"], all_fit["sigma"]),
                color="tab:gray",
                alpha=0.95,
                lw=2.2,
                label="Gaussian fit: test set",
            )
        if np.isfinite(cut_fit["sigma"]) and cut_fit["sigma"] > 0:
            ax.plot(
                x_fit,
                _gaussian(x_fit, cut_fit["amplitude"], cut_fit["mean"], cut_fit["sigma"]),
                color="tab:blue",
                alpha=0.95,
                lw=2.2,
                label=rf"Gaussian fit: $\sigma_{{{name}}} < \tilde{{\sigma}}_{{{name}}}$",
            )
        ax.axvline(0.0, color="black", ls="--", lw=1)
        ax.set_xlabel(f"{name}_true - {name}_pred (mm)", loc="right")
        ax.set_ylabel("Fraction of test set", loc="top")

        all_stats = _residual_summary(finite_all)
        cut_stats = _residual_summary(cut_residual)
        all_fwhm = all_fit["fwhm"]
        cut_fwhm = cut_fit["fwhm"]
        improvement = 100.0 * (1.0 - cut_fwhm / all_fwhm) if all_fwhm > 0 else np.nan

        metrics[name] = {
            "all": all_stats,
            "sigma_cut": cut_stats,
            "all_gaussian_fit": all_fit,
            "sigma_cut_gaussian_fit": cut_fit,
            "median_uncertainty_cut_mm": float(uncertainty_cut_mm),
            "selected_fraction": float(cut_stats["n"] / max(all_stats["n"], 1)),
            "fwhm_improvement_factor": float(all_fwhm / cut_fwhm) if cut_fwhm > 0 else np.nan,
            "fwhm_percent_improvement": float(improvement),
        }

        txt = (
            rf"$\mathrm{{FWHM}}_{{\mathrm{{all}}}} = {all_fwhm:.3g}\,\mathrm{{mm}}$"
            "\n"
            rf"$\mathrm{{FWHM}}_{{\sigma < \mathrm{{median}}}} = {cut_fwhm:.3g}\,\mathrm{{mm}}$"
            "\n"
            rf"$\Delta\mathrm{{FWHM}} = {improvement:.1f}\%$"
        )
        ax.text(
            0.98,
            0.95,
            txt,
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=8,
        )
        ax.legend(fontsize=8, loc="best")

    _save_paper_figure(fig, outdir / "residuals-with-predicted-uncertainty-cut.png")

    return metrics


def plot_residual_fwhm_vs_true_z(
    y_true,
    y_pred,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    predicted_uncertainties=None,
    n_z_bins=8,
    min_events_per_bin=30,
):
    """Plot Gaussian-fit residual FWHM for x/y/z in slices of true z."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    output_names = tuple(output_names)
    spatial_indices = tuple(spatial_indices)

    z_index = spatial_indices[2]
    true_z = y_true[:, z_index]
    residual = y_true - y_pred
    predicted_uncertainties = None if predicted_uncertainties is None else np.asarray(predicted_uncertainties)

    valid_z = np.isfinite(true_z)
    if valid_z.sum() < min_events_per_bin:
        print("Skipping residual FWHM vs true z: too few finite z values")
        return {}

    z_min = float(np.nanmin(true_z[valid_z]))
    z_max = float(np.nanmax(true_z[valid_z]))
    if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
        print("Skipping residual FWHM vs true z: invalid z range")
        return {}

    edges = np.linspace(z_min, z_max, n_z_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    metrics = {
        "z_bin_edges": edges.tolist(),
        "z_bin_centers": centers.tolist(),
        "min_events_per_bin": int(min_events_per_bin),
        "slice_fit_plot_dir": "residual-fits-vs-true-z",
        "axes": {},
    }

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    fig_mean, ax_mean = plt.subplots(figsize=(7.2, 4.8))
    low_zsigma_mask = None
    if (
        predicted_uncertainties is not None
        and predicted_uncertainties.shape[0] == y_true.shape[0]
        and predicted_uncertainties.shape[1] > z_index
    ):
        sigma_z = predicted_uncertainties[:, z_index]
        finite_sigma_z = np.isfinite(sigma_z)
        if np.any(finite_sigma_z):
            median_sigma_z = float(np.median(sigma_z[finite_sigma_z]))
            low_zsigma_mask = finite_sigma_z & (sigma_z < median_sigma_z)
            metrics["low_zsigma_selection"] = {
                "criterion": "sigma_z < median(sigma_z)",
                "median_sigma_z_mm": median_sigma_z,
                "selected_fraction": float(low_zsigma_mask.sum() / len(low_zsigma_mask)),
            }

    fig_fwhm_low = None
    ax_fwhm_low = None
    fig_fwhm_compare = None
    axes_fwhm_compare = None
    fig_mean_low = None
    ax_mean_low = None
    if low_zsigma_mask is not None:
        fig_fwhm_low, ax_fwhm_low = plt.subplots(figsize=(7.2, 4.8))
        fig_fwhm_compare, axes_fwhm_compare = plt.subplots(
            1,
            len(spatial_indices),
            figsize=(5.0 * len(spatial_indices), 4.2),
            squeeze=False,
        )
        axes_fwhm_compare = axes_fwhm_compare.reshape(-1)
        fig_mean_low, ax_mean_low = plt.subplots(figsize=(7.2, 4.8))
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for col, (color, idx) in enumerate(zip(colors, spatial_indices)):
        name = output_names[idx]
        fwhm_values = []
        mean_values = []
        low_zsigma_fwhm_values = []
        low_zsigma_mean_values = []
        n_values = []
        low_zsigma_n_values = []
        fit_ok_values = []
        low_zsigma_fit_ok_values = []

        for lo, hi in zip(edges[:-1], edges[1:]):
            in_bin = valid_z & (true_z >= lo) & (true_z < hi)
            if hi == edges[-1]:
                in_bin = valid_z & (true_z >= lo) & (true_z <= hi)

            axis_residual = residual[in_bin, idx]
            axis_residual = axis_residual[np.isfinite(axis_residual)]
            n = int(len(axis_residual))
            n_values.append(n)

            if n < min_events_per_bin:
                fwhm_values.append(np.nan)
                mean_values.append(np.nan)
                fit_ok_values.append(False)
                if low_zsigma_mask is not None:
                    selected = in_bin & low_zsigma_mask
                    selected_residual = residual[selected, idx]
                    selected_residual = selected_residual[np.isfinite(selected_residual)]
                    low_zsigma_n_values.append(int(len(selected_residual)))
                else:
                    low_zsigma_n_values.append(0)
                low_zsigma_fwhm_values.append(np.nan)
                low_zsigma_mean_values.append(np.nan)
                low_zsigma_fit_ok_values.append(False)
                continue

            fit = _fit_gaussian_to_histogram(axis_residual, bins=40)
            fwhm_values.append(float(fit["fwhm"]))
            mean_values.append(float(fit["mean"]))
            fit_ok_values.append(bool(fit["fit_ok"]))

            if low_zsigma_mask is not None:
                selected = in_bin & low_zsigma_mask
                selected_residual = residual[selected, idx]
                selected_residual = selected_residual[np.isfinite(selected_residual)]
                n_selected = int(len(selected_residual))
                low_zsigma_n_values.append(n_selected)
                if n_selected < min_events_per_bin:
                    low_zsigma_fwhm_values.append(np.nan)
                    low_zsigma_mean_values.append(np.nan)
                    low_zsigma_fit_ok_values.append(False)
                else:
                    selected_fit = _fit_gaussian_to_histogram(selected_residual, bins=40)
                    low_zsigma_fwhm_values.append(float(selected_fit["fwhm"]))
                    low_zsigma_mean_values.append(float(selected_fit["mean"]))
                    low_zsigma_fit_ok_values.append(bool(selected_fit["fit_ok"]))
            else:
                low_zsigma_fwhm_values.append(np.nan)
                low_zsigma_mean_values.append(np.nan)
                low_zsigma_n_values.append(0)
                low_zsigma_fit_ok_values.append(False)

        ax.plot(
            centers,
            fwhm_values,
            marker="o",
            ms=4,
            lw=1.8,
            color=color,
            label=f"{name} residual FWHM",
        )
        if ax_fwhm_low is not None:
            ax_fwhm_low.plot(
                centers,
                low_zsigma_fwhm_values,
                marker="o",
                ms=4,
                lw=1.8,
                color=color,
                label=f"{name} residual FWHM",
            )
        if axes_fwhm_compare is not None:
            ax_compare = axes_fwhm_compare[col]
            ax_compare.plot(
                centers,
                fwhm_values,
                marker="o",
                ms=4,
                lw=1.8,
                color=color,
                label="All events",
            )
            ax_compare.plot(
                centers,
                low_zsigma_fwhm_values,
                marker="s",
                ms=4,
                lw=1.8,
                ls="--",
                color=color,
                label=rf"$\sigma_z < \tilde{{\sigma}}_z$",
            )
            ax_compare.set_xlabel("True z position (mm)", loc="right")
            ax_compare.set_ylabel("Gaussian-fit residual FWHM (mm)", loc="top")
            ax_compare.legend(fontsize=8, loc="best")
        ax_mean.plot(
            centers,
            mean_values,
            marker="o",
            ms=4,
            lw=1.8,
            color=color,
            label=f"{name} residual mean",
        )
        if ax_mean_low is not None:
            ax_mean_low.plot(
                centers,
                low_zsigma_mean_values,
                marker="o",
                ms=4,
                lw=1.8,
                color=color,
                label=f"{name} residual mean",
            )

        metrics["axes"][name] = {
            "fwhm_mm": [float(v) if np.isfinite(v) else np.nan for v in fwhm_values],
            "mean_mm": [float(v) if np.isfinite(v) else np.nan for v in mean_values],
            "n_events": n_values,
            "fit_ok": fit_ok_values,
            "low_zsigma_fwhm_mm": [float(v) if np.isfinite(v) else np.nan for v in low_zsigma_fwhm_values],
            "low_zsigma_mean_mm": [float(v) if np.isfinite(v) else np.nan for v in low_zsigma_mean_values],
            "low_zsigma_n_events": low_zsigma_n_values,
            "low_zsigma_fit_ok": low_zsigma_fit_ok_values,
        }

    ax.set_xlabel("True z position (mm)", loc="right")
    ax.set_ylabel("Gaussian-fit residual FWHM (mm)", loc="top")
    ax.legend(fontsize=8, loc="best")
    _save_paper_figure(fig, outdir / "residual-fwhm-vs-true-z.png")

    if fig_fwhm_low is not None:
        ax_fwhm_low.set_xlabel("True z position (mm)", loc="right")
        ax_fwhm_low.set_ylabel("Gaussian-fit residual FWHM (mm)", loc="top")
        ax_fwhm_low.legend(fontsize=8, loc="best")
        _save_paper_figure(fig_fwhm_low, outdir / "residual-fwhm-vs-true-z-low-zsigma.png")

    if fig_fwhm_compare is not None:
        _save_paper_figure(fig_fwhm_compare, outdir / "residual-fwhm-vs-true-z-comparison.png")

    ax_mean.axhline(0.0, color="black", ls="--", lw=1)
    ax_mean.set_xlabel("True z position (mm)", loc="right")
    ax_mean.set_ylabel("Gaussian-fit residual mean (mm)", loc="top")
    ax_mean.legend(fontsize=8, loc="best")
    _save_paper_figure(fig_mean, outdir / "residual-mean-vs-true-z.png")

    if fig_mean_low is not None:
        ax_mean_low.axhline(0.0, color="black", ls="--", lw=1)
        ax_mean_low.set_xlabel("True z position (mm)", loc="right")
        ax_mean_low.set_ylabel("Gaussian-fit residual mean (mm)", loc="top")
        ax_mean_low.legend(fontsize=8, loc="best")
        _save_paper_figure(fig_mean_low, outdir / "residual-mean-vs-true-z-low-zsigma.png")

    slice_dir = outdir / "residual-fits-vs-true-z"
    slice_dir.mkdir(parents=True, exist_ok=True)
    def _plot_residual_slice_panel(ax_slice, axis_residual, color, name, title):
        axis_residual = axis_residual[np.isfinite(axis_residual)]
        n_events = int(len(axis_residual))

        if n_events == 0:
            ax_slice.text(0.5, 0.5, "No events", transform=ax_slice.transAxes, ha="center", va="center")
            ax_slice.set_axis_off()
            return

        r_lo, r_hi = np.nanpercentile(axis_residual, [0.5, 99.5])
        if not np.isfinite(r_lo) or not np.isfinite(r_hi) or r_lo == r_hi:
            r_lo, r_hi = float(np.nanmin(axis_residual)), float(np.nanmax(axis_residual))
        if r_lo == r_hi:
            r_lo -= 0.5
            r_hi += 0.5

        bins = np.linspace(r_lo, r_hi, 40)
        ax_slice.hist(axis_residual, bins=bins, density=True, alpha=0.45, color=color, label="residuals")

        fit = _fit_gaussian_pdf(axis_residual, bins)
        if np.isfinite(fit["sigma"]) and fit["sigma"] > 0:
            x_fit = np.linspace(r_lo, r_hi, 300)
            ax_slice.plot(
                x_fit,
                _gaussian(x_fit, fit["amplitude"], fit["mean"], fit["sigma"]),
                color=color,
                alpha=0.95,
                lw=2.2,
                label="Gaussian fit",
            )
        ax_slice.axvline(0.0, color="black", ls="--", lw=1)
        ax_slice.set_xlabel(f"{name}_true - {name}_pred (mm)", loc="right")
        ax_slice.set_ylabel("Density", loc="top")
        ax_slice.text(
            0.98,
            0.95,
            (
                f"n = {n_events}\n"
                rf"$\mu = {fit['mean']:.3g}\,\mathrm{{mm}}$"
                "\n"
                rf"$\mathrm{{FWHM}} = {fit['fwhm']:.3g}\,\mathrm{{mm}}$"
                "\n"
                f"fit ok = {fit['fit_ok']}"
            ),
            transform=ax_slice.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=8,
        )
        ax_slice.legend(fontsize=8, loc="best")

    for bin_index, (lo, hi, z_center) in enumerate(zip(edges[:-1], edges[1:], centers)):
        in_bin = valid_z & (true_z >= lo) & (true_z < hi)
        if hi == edges[-1]:
            in_bin = valid_z & (true_z >= lo) & (true_z <= hi)

        n_rows = 2 if low_zsigma_mask is not None else 1
        fig, axes = plt.subplots(
            n_rows,
            len(spatial_indices),
            figsize=(5.0 * len(spatial_indices), 4.0 * n_rows),
            squeeze=False,
        )

        for col, (color, idx) in enumerate(zip(colors, spatial_indices)):
            name = output_names[idx]
            axis_residual = residual[in_bin, idx]
            _plot_residual_slice_panel(axes[0, col], axis_residual, color, name, f"All events: {name}")

            if low_zsigma_mask is not None:
                selected = in_bin & low_zsigma_mask
                selected_residual = residual[selected, idx]
                _plot_residual_slice_panel(
                    axes[1, col],
                    selected_residual,
                    color,
                    name,
                    rf"$\sigma_z < \tilde{{\sigma}}_z$: {name}",
                )

        _save_paper_figure(fig, slice_dir / f"residual-fits-zbin-{bin_index:02d}.png")

    return metrics


def plot_true_predicted_coordinate_distributions(
    y_true,
    y_pred,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    predicted_uncertainties=None,
):
    """Overlay true and predicted coordinate distributions for each spatial axis."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    predicted_uncertainties = None if predicted_uncertainties is None else np.asarray(predicted_uncertainties)
    output_names = tuple(output_names)
    spatial_indices = tuple(spatial_indices)

    low_zsigma_mask = None
    if len(spatial_indices) >= 3:
        z_index = spatial_indices[2]
        if (
            predicted_uncertainties is not None
            and predicted_uncertainties.shape[0] == y_true.shape[0]
            and predicted_uncertainties.shape[1] > z_index
        ):
            sigma_z = predicted_uncertainties[:, z_index]
            finite_sigma_z = np.isfinite(sigma_z)
            if np.any(finite_sigma_z):
                low_zsigma_mask = finite_sigma_z & (sigma_z < np.median(sigma_z[finite_sigma_z]))

    n_rows = 2 if low_zsigma_mask is not None else 1
    fig, axes = plt.subplots(
        n_rows,
        len(spatial_indices),
        figsize=(5.0 * len(spatial_indices), 4.0 * n_rows),
        squeeze=False,
    )
    metrics = {}

    def _draw_true_pred_hist(ax, true_values, pred_values, bins, name, title):
        finite = np.isfinite(true_values) & np.isfinite(pred_values)
        true_values = true_values[finite]
        pred_values = pred_values[finite]

        if len(true_values) == 0:
            ax.text(0.5, 0.5, "No finite values", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            return true_values, pred_values

        ax.hist(
            true_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color="tab:blue",
            label=f"true {name}",
        )
        ax.hist(
            pred_values,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.35,
            color="tab:orange",
            label=f"predicted {name}",
        )
        ax.set_xlabel(f"{name} position (mm)", loc="right")
        ax.set_ylabel("Density", loc="top")
        ax.legend(fontsize=8, loc="best")
        return true_values, pred_values

    for col, idx in enumerate(spatial_indices):
        name = output_names[idx]
        true_values = y_true[:, idx]
        pred_values = y_pred[:, idx]
        finite = np.isfinite(true_values) & np.isfinite(pred_values)
        finite_true_values = true_values[finite]
        finite_pred_values = pred_values[finite]

        if len(finite_true_values) == 0:
            axes[0, col].text(0.5, 0.5, "No finite values", transform=axes[0, col].transAxes, ha="center", va="center")
            axes[0, col].set_axis_off()
            if low_zsigma_mask is not None:
                axes[1, col].set_axis_off()
            continue

        lo = float(np.nanmin(np.concatenate([finite_true_values, finite_pred_values])))
        hi = float(np.nanmax(np.concatenate([finite_true_values, finite_pred_values])))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        bins = np.linspace(lo, hi, 60)

        true_all, pred_all = _draw_true_pred_hist(
            axes[0, col],
            true_values,
            pred_values,
            bins,
            name,
            f"All events: {name}",
        )

        metrics[name] = {
            "n": int(len(true_all)),
            "true_mean": float(np.mean(true_all)),
            "pred_mean": float(np.mean(pred_all)),
            "true_std": float(np.std(true_all, ddof=1)) if len(true_all) > 1 else np.nan,
            "pred_std": float(np.std(pred_all, ddof=1)) if len(pred_all) > 1 else np.nan,
        }

        if low_zsigma_mask is not None:
            true_low, pred_low = _draw_true_pred_hist(
                axes[1, col],
                true_values[low_zsigma_mask],
                pred_values[low_zsigma_mask],
                bins,
                name,
                rf"$\sigma_z < \tilde{{\sigma}}_z$: {name}",
            )
            metrics[name].update(
                {
                    "low_zsigma_n": int(len(true_low)),
                    "low_zsigma_true_mean": float(np.mean(true_low)) if len(true_low) > 0 else np.nan,
                    "low_zsigma_pred_mean": float(np.mean(pred_low)) if len(pred_low) > 0 else np.nan,
                    "low_zsigma_true_std": float(np.std(true_low, ddof=1)) if len(true_low) > 1 else np.nan,
                    "low_zsigma_pred_std": float(np.std(pred_low, ddof=1)) if len(pred_low) > 1 else np.nan,
                }
            )

    _save_paper_figure(fig, outdir / "true-vs-predicted-coordinate-distributions.png")

    return metrics


def _weighted_hist_quantiles(centers, counts, probabilities):
    centers = np.asarray(centers, dtype=float)
    counts = np.asarray(counts, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    total = np.sum(counts)
    if total <= 0:
        return np.full_like(probabilities, np.nan, dtype=float)
    cdf = np.cumsum(counts) / total
    return np.interp(probabilities, cdf, centers, left=centers[0], right=centers[-1])


def _fit_binned_gmm_1d(counts, edges, n_components=4, max_iter=250, tol=1e-7):
    """Fit a 1D GMM to histogram counts with EM.

    This is intended for diagnostics. Fitting to binned counts is much faster
    than fitting to every event while preserving the distribution shape visible
    in these plots.
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    centers = 0.5 * (edges[:-1] + edges[1:])
    positive = np.isfinite(counts) & (counts > 0) & np.isfinite(centers)
    centers = centers[positive]
    counts = counts[positive]

    n_total = float(np.sum(counts))
    if n_total <= 0 or len(centers) < 2:
        return {"fit_ok": False, "reason": "too few populated bins"}

    n_components = int(max(1, min(n_components, len(centers))))
    bin_width = float(np.nanmedian(np.diff(edges))) if len(edges) > 2 else 1.0
    min_sigma = max(0.5 * abs(bin_width), 1e-3)
    full_mean = float(np.average(centers, weights=counts))
    full_var = float(np.average((centers - full_mean) ** 2, weights=counts))
    full_sigma = max(np.sqrt(max(full_var, 0.0)), min_sigma)

    probabilities = (np.arange(n_components, dtype=float) + 0.5) / n_components
    means = _weighted_hist_quantiles(centers, counts, probabilities)
    means = np.where(np.isfinite(means), means, full_mean)
    sigmas = np.full(n_components, max(full_sigma / np.sqrt(n_components), min_sigma))
    weights = np.full(n_components, 1.0 / n_components)

    previous_log_likelihood = -np.inf
    fit_ok = False
    reason = "maximum iterations reached"

    for iteration in range(int(max_iter)):
        pdf = _normal_pdf_values(centers, means, sigmas)
        weighted_pdf = np.maximum(pdf * weights[None, :], LOSS_EPS)
        mixture_pdf = np.sum(weighted_pdf, axis=1)
        log_likelihood = float(np.sum(counts * np.log(np.maximum(mixture_pdf, LOSS_EPS))))

        responsibilities = weighted_pdf / np.maximum(mixture_pdf[:, None], LOSS_EPS)
        weighted_responsibilities = counts[:, None] * responsibilities
        component_counts = np.sum(weighted_responsibilities, axis=0)
        tiny = component_counts <= max(LOSS_EPS, 1e-8 * n_total)
        safe_component_counts = np.where(tiny, 1.0, component_counts)
        new_means = (
            np.sum(weighted_responsibilities * centers[:, None], axis=0)
            / safe_component_counts
        )
        variances = (
            np.sum(weighted_responsibilities * (centers[:, None] - new_means[None, :]) ** 2, axis=0)
            / safe_component_counts
        )
        new_sigmas = np.maximum(np.sqrt(np.maximum(variances, 0.0)), min_sigma)

        if np.any(tiny):
            replacement_means = _weighted_hist_quantiles(
                centers,
                counts,
                (np.arange(np.sum(tiny), dtype=float) + 0.5) / max(np.sum(tiny), 1),
            )
            new_means[tiny] = replacement_means
            new_sigmas[tiny] = full_sigma
            component_counts[tiny] = 1e-8 * n_total

        weights = component_counts / np.sum(component_counts)
        means = new_means
        sigmas = new_sigmas

        order = np.argsort(means)
        weights = weights[order]
        means = means[order]
        sigmas = sigmas[order]

        if np.isfinite(previous_log_likelihood):
            scale = max(abs(previous_log_likelihood), 1.0)
            if abs(log_likelihood - previous_log_likelihood) / scale < tol:
                fit_ok = True
                reason = "converged"
                break
        previous_log_likelihood = log_likelihood

    weights = np.maximum(weights, LOSS_EPS)
    weights = weights / np.sum(weights)

    return {
        "fit_ok": bool(fit_ok),
        "reason": reason,
        "n_components": int(n_components),
        "weights": weights.tolist(),
        "means": means.tolist(),
        "sigmas": sigmas.tolist(),
        "log_likelihood": float(previous_log_likelihood),
    }


def _gmm_pdf_from_fit(x, fit):
    weights = np.asarray(fit.get("weights", []), dtype=float)
    means = np.asarray(fit.get("means", []), dtype=float)
    sigmas = np.asarray(fit.get("sigmas", []), dtype=float)
    if len(weights) == 0:
        return np.full_like(np.asarray(x, dtype=float), np.nan)
    pdf_components = _normal_pdf_values(np.asarray(x, dtype=float), means, sigmas)
    return np.sum(pdf_components * weights[None, :], axis=1)


def _gmm_summary_from_fit(fit, z_lo, z_hi, grid_size=2400):
    weights = np.asarray(fit.get("weights", []), dtype=float)
    means = np.asarray(fit.get("means", []), dtype=float)
    if len(weights) == 0:
        return {"mean": np.nan, "median": np.nan, "mode": np.nan}

    grid = np.linspace(float(z_lo), float(z_hi), int(grid_size))
    pdf = _gmm_pdf_from_fit(grid, fit)
    finite = np.isfinite(pdf)
    if not np.any(finite) or np.nanmax(pdf) <= 0:
        return {"mean": float(np.sum(weights * means)), "median": np.nan, "mode": np.nan}

    mode = float(grid[int(np.nanargmax(pdf))])
    dx = np.diff(grid)
    cumulative = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * dx)])
    if cumulative[-1] > 0:
        cumulative = cumulative / cumulative[-1]
        median = float(np.interp(0.5, cumulative, grid))
    else:
        median = np.nan

    return {
        "mean": float(np.sum(weights * means)),
        "median": median,
        "mode": mode,
    }


def _draw_gmm_fit_and_estimators(ax, fit, summary, z_lo, z_hi):
    if not fit.get("weights"):
        return

    x_fit = np.linspace(float(z_lo), float(z_hi), 800)
    y_fit = _gmm_pdf_from_fit(x_fit, fit)
    if np.any(np.isfinite(y_fit)):
        ax.plot(x_fit, y_fit, color="black", lw=2.0, label=f"{fit['n_components']}-component GMM")

    estimator_styles = [
        ("mean", "tab:purple", "--", r"GMM mean"),
        ("median", "tab:green", "-", r"GMM median"),
        ("mode", "tab:red", "-.", r"GMM mode"),
    ]
    for key, color, linestyle, label in estimator_styles:
        value = summary.get(key, np.nan)
        if np.isfinite(value):
            ax.axvline(value, color=color, lw=1.8, ls=linestyle, label=label)


def plot_true_z_by_predicted_z_bins(
    y_true,
    y_pred,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    n_pred_z_bins=8,
    hist_bins=60,
    min_events_per_bin=20,
    gmm_components=4,
):
    """Plot empirical true-z distributions in quantile bins of predicted z.

    This is a direct diagnostic for the learned conditional density: if events
    with similar predicted z still have a broad, skewed, or multi-modal true-z
    distribution, a point prediction or single Gaussian p(z|X) is intrinsically
    limited in that region of feature space.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    output_names = tuple(output_names)

    if "z" in output_names:
        z_index = output_names.index("z")
    else:
        z_index = spatial_indices[2]

    true_z = y_true[:, z_index]
    pred_z = y_pred[:, z_index]
    finite = np.isfinite(true_z) & np.isfinite(pred_z)
    true_z = true_z[finite]
    pred_z = pred_z[finite]

    if len(true_z) < max(min_events_per_bin, 2 * n_pred_z_bins):
        print("Skipping true-z by predicted-z bins: too few finite events")
        return {"n": int(len(true_z)), "skipped": True}

    n_pred_z_bins = int(max(2, n_pred_z_bins))
    quantiles = np.linspace(0.0, 1.0, n_pred_z_bins + 1)
    edges = np.quantile(pred_z, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.linspace(float(np.nanmin(pred_z)), float(np.nanmax(pred_z)), n_pred_z_bins + 1)
        edges = np.unique(edges)
    if len(edges) < 3:
        print("Skipping true-z by predicted-z bins: invalid predicted-z bin edges")
        return {"n": int(len(true_z)), "skipped": True}

    z_lo = float(np.nanmin(true_z))
    z_hi = float(np.nanmax(true_z))
    if not np.isfinite(z_lo) or not np.isfinite(z_hi) or z_lo == z_hi:
        print("Skipping true-z by predicted-z bins: invalid z range")
        return {"n": int(len(true_z)), "skipped": True}

    bins = np.linspace(z_lo, z_hi, hist_bins + 1)
    diagnostic_dir = outdir / "true-z-by-predicted-z-bin"
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n": int(len(true_z)),
        "binning": "predicted-z quantiles",
        "pred_z_bin_edges": edges.tolist(),
        "min_events_per_bin": int(min_events_per_bin),
        "gmm_components": int(gmm_components),
        "bins": [],
    }

    valid_bin_records = []
    for bin_index, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if bin_index == len(edges) - 2:
            in_bin = (pred_z >= lo) & (pred_z <= hi)
        else:
            in_bin = (pred_z >= lo) & (pred_z < hi)

        z_selected = true_z[in_bin]
        pred_selected = pred_z[in_bin]
        n_selected = int(len(z_selected))
        record = {
            "bin_index": int(bin_index),
            "pred_z_low_mm": float(lo),
            "pred_z_high_mm": float(hi),
            "n": n_selected,
            "true_z_mean_mm": float(np.mean(z_selected)) if n_selected > 0 else np.nan,
            "true_z_median_mm": float(np.median(z_selected)) if n_selected > 0 else np.nan,
            "true_z_std_mm": float(np.std(z_selected, ddof=1)) if n_selected > 1 else np.nan,
            "pred_z_mean_mm": float(np.mean(pred_selected)) if n_selected > 0 else np.nan,
            "pred_z_median_mm": float(np.median(pred_selected)) if n_selected > 0 else np.nan,
        }
        metrics["bins"].append(record)

        if n_selected < min_events_per_bin:
            continue

        gmm_counts, _ = np.histogram(z_selected, bins=bins)
        gmm_fit = _fit_binned_gmm_1d(gmm_counts, bins, n_components=gmm_components)
        gmm_summary = _gmm_summary_from_fit(gmm_fit, z_lo, z_hi)
        record["gmm_fit"] = {
            "fit_ok": bool(gmm_fit.get("fit_ok", False)),
            "reason": gmm_fit.get("reason", ""),
            "n_components": int(gmm_fit.get("n_components", 0)),
            "weights": gmm_fit.get("weights", []),
            "means_mm": gmm_fit.get("means", []),
            "sigmas_mm": gmm_fit.get("sigmas", []),
            "mean_mm": float(gmm_summary["mean"]) if np.isfinite(gmm_summary["mean"]) else np.nan,
            "median_mm": float(gmm_summary["median"]) if np.isfinite(gmm_summary["median"]) else np.nan,
            "mode_mm": float(gmm_summary["mode"]) if np.isfinite(gmm_summary["mode"]) else np.nan,
        }

        valid_bin_records.append((bin_index, lo, hi, z_selected, pred_selected, record, gmm_fit, gmm_summary))

        fig, ax = plt.subplots(figsize=VALIDATION_FIGSIZE)
        ax.hist(
            z_selected,
            bins=bins,
            density=True,
            histtype="stepfilled",
            alpha=0.72,
            color="tab:blue",
        )
        ax.axvspan(lo, hi, color="tab:orange", alpha=0.18, label="predicted-z bin")
        _draw_gmm_fit_and_estimators(ax, gmm_fit, gmm_summary, z_lo, z_hi)
        ax.axvline(np.median(z_selected), color="tab:blue", lw=1.7, ls=":", label="empirical true-z median")
        ax.axvline(np.median(pred_selected), color="tab:orange", lw=1.7, ls=":", label="pred-z median")
        ax.set_xlim(z_lo, z_hi)
        ax.set_xlabel("True z (mm)", loc="right")
        ax.set_ylabel("Density", loc="top")
        gmm_text = (
            rf"GMM mean $= {gmm_summary['mean']:.2f}\,\mathrm{{mm}}$"
            "\n"
            rf"GMM median $= {gmm_summary['median']:.2f}\,\mathrm{{mm}}$"
            "\n"
            rf"GMM mode $= {gmm_summary['mode']:.2f}\,\mathrm{{mm}}$"
        )
        ax.text(
            0.97,
            0.94,
            (
                rf"${lo:.2f} < z_{{\mathrm{{pred}}}} < {hi:.2f}\,\mathrm{{mm}}$"
                "\n"
                f"n = {n_selected:,}"
                "\n"
                f"{gmm_text}"
            ),
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
            fontsize=PAPER_SMALL_TEXT_SIZE,
        )
        ax.legend(fontsize=PAPER_SMALL_TEXT_SIZE, loc="upper left")
        _save_paper_figure(fig, diagnostic_dir / f"true-z-for-pred-z-bin-{bin_index:02d}.png")

    if valid_bin_records:
        n_cols = min(4, len(valid_bin_records))
        n_rows = int(np.ceil(len(valid_bin_records) / n_cols))
        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(4.4 * n_cols, 3.5 * n_rows),
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        axes_flat = axes.ravel()
        for ax, (bin_index, lo, hi, z_selected, pred_selected, record, gmm_fit, gmm_summary) in zip(axes_flat, valid_bin_records):
            ax.hist(
                z_selected,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.72,
                color="tab:blue",
            )
            ax.axvspan(lo, hi, color="tab:orange", alpha=0.18)
            _draw_gmm_fit_and_estimators(ax, gmm_fit, gmm_summary, z_lo, z_hi)
            ax.set_xlim(z_lo, z_hi)
            ax.text(
                0.96,
                0.92,
                (
                    rf"$z_{{\mathrm{{pred}}}}\in[{lo:.2f},{hi:.2f}]$"
                    "\n"
                    f"n={len(z_selected):,}"
                    "\n"
                    rf"$\mu_{{\mathrm{{GMM}}}}={gmm_summary['mean']:.2f}$"
                    "\n"
                    rf"$\tilde{{z}}_{{\mathrm{{GMM}}}}={gmm_summary['median']:.2f}$"
                    "\n"
                    rf"$z_{{\mathrm{{mode}}}}={gmm_summary['mode']:.2f}$"
                ),
                transform=ax.transAxes,
                ha="right",
                va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.78),
                fontsize=PAPER_SMALL_TEXT_SIZE,
            )
            ax.set_xlabel("True z (mm)", loc="right")
            ax.set_ylabel("Density", loc="top")

        for ax in axes_flat[len(valid_bin_records):]:
            ax.set_axis_off()
        _save_paper_figure(fig, diagnostic_dir / "true-z-by-predicted-z-bin-summary.png")

    return metrics


def evaluate_model_performance(
    y_true,
    y_pred,
    outdir,
    output_names,
    spatial_indices=(0, 1, 2),
    predicted_uncertainties=None,
    metadata=None,
    process_column="process",
    pixel_geometry=None,
    min_events_per_pixel=5,
    n_z_resolution_bins=8,
    incident_energy_window_mev=0.001,
    source_radius_mm=13.0,
    source_distance_cm=20.0,
    cone_margin=0.3,
    cone_half_angle_deg=None,
    cone_angle_samples=96,
    pe_spectrum_mode_min=PE_SPECTRUM_MODE_MIN,
    pe_spectrum_fit_fraction=PE_SPECTRUM_FIT_FRACTION,
    pe_spectrum_fit_low_fraction=None,
    pe_spectrum_fit_high_fraction=None,
    pe_spectrum_fit_model="gauss-linear",
    max_pe_feature_positions=None,
    make_per_pixel_plots=True,
    n_pred_z_condition_bins=8,
    n_pred_z_condition_gmm_components=4,
    z_gmm_point_estimator="mean",
    z_gmm_mode_grid_size=801,
):
    """Create standard performance plots and return scalar metrics.

    This function is deliberately model-agnostic. It only needs true targets,
    predicted targets, output names, and optional metadata for process/pixel
    diagnostics.
    """
    outdir = _as_path(outdir)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    output_names = tuple(output_names)

    z_point_prediction = "input"
    if metadata is not None and "z" in output_names:
        z_index = output_names.index("z")
        finite_z = y_true[:, z_index][np.isfinite(y_true[:, z_index])]
        if pixel_geometry is not None and "pixel_length_mm" in pixel_geometry:
            half_length = 0.5 * float(pixel_geometry["pixel_length_mm"])
            z_bounds = (-half_length, half_length)
        elif len(finite_z) > 0:
            z_bounds = (float(np.nanmin(finite_z)), float(np.nanmax(finite_z)))
        else:
            z_bounds = None

        y_pred, used_z_gmm_estimator = use_z_gmm_point_prediction(
            y_pred,
            output_names,
            metadata,
            estimator=z_gmm_point_estimator,
            z_bounds=z_bounds,
            mode_grid_size=z_gmm_mode_grid_size,
        )
        if used_z_gmm_estimator:
            z_point_prediction = f"gmm_{z_gmm_point_estimator}"
            print(f"Using z GMM {z_gmm_point_estimator} for z evaluation")

    residual = y_true - y_pred
    spatial_residual = residual[:, list(spatial_indices)]
    spatial_error = np.linalg.norm(spatial_residual, axis=1)

    spatial_names = [output_names[i] for i in spatial_indices]
    spatial_mae = np.mean(np.abs(spatial_residual), axis=0)
    spatial_rmse = np.sqrt(np.mean(spatial_residual ** 2, axis=0))

    print(f"Test mean 3D error: {np.mean(spatial_error):.4f} mm")

    plot_prediction_scatter(y_true, y_pred, output_names, outdir)

    pull_metrics = {}
    for idx, name in enumerate(output_names):
        if idx in spatial_indices:
            unit = "mm"
        elif name in {"theta", "angle", "incident_angle"}:
            unit = "deg"
        else:
            unit = ""
        xlabel = f"{name}_true - {name}_pred" + (f" ({unit})" if unit else "")
        pull_metrics[name] = plot_residual_histogram(
            residual[:, idx],
            outdir / f"pull-{name}.png",
            f"{name} residuals (test set)",
            xlabel,
        )

    delta_ke_pull_metrics = plot_spatial_pulls_by_gamma_delta_ke(
        y_true,
        y_pred,
        metadata,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
    )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spatial_error, bins=50, alpha=0.85)
    ax.set_xlabel("3D position error |r_true - r_pred| (mm)", loc="right")
    ax.set_ylabel("Counts", loc="top")
    _save_paper_figure(fig, outdir / "position-error-3d.png")

    metrics = {
        "mean_3d_error_mm": float(np.mean(spatial_error)),
        "median_3d_error_mm": float(np.median(spatial_error)),
        "p68_3d_error_mm": float(np.percentile(spatial_error, 68)),
        "p95_3d_error_mm": float(np.percentile(spatial_error, 95)),
        "spatial_mae_mm": {name: float(spatial_mae[i]) for i, name in enumerate(spatial_names)},
        "spatial_rmse_mm": {name: float(spatial_rmse[i]) for i, name in enumerate(spatial_names)},
        "pull": pull_metrics,
        "z_point_prediction": z_point_prediction,
    }
    if delta_ke_pull_metrics:
        metrics["pull_by_gamma_delta_ke"] = delta_ke_pull_metrics

    uncertainty_metrics = plot_uncertainty_diagnostics(
        y_true,
        y_pred,
        predicted_uncertainties,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
    )
    if uncertainty_metrics:
        metrics["uncertainty"] = uncertainty_metrics

    uncertainty_cut_metrics = plot_residuals_with_uncertainty_cut(
        y_true,
        y_pred,
        predicted_uncertainties,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
    )
    if uncertainty_cut_metrics:
        metrics["residuals_with_uncertainty_cut"] = uncertainty_cut_metrics

    uncertainty_tradeoff_metrics = plot_uncertainty_resolution_sensitivity_tradeoff(
        y_true,
        y_pred,
        predicted_uncertainties,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
    )
    if uncertainty_tradeoff_metrics:
        metrics["uncertainty_resolution_sensitivity_tradeoff"] = uncertainty_tradeoff_metrics

    fwhm_vs_z_metrics = plot_residual_fwhm_vs_true_z(
        y_true,
        y_pred,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        predicted_uncertainties=predicted_uncertainties,
        n_z_bins=n_z_resolution_bins,
    )
    if fwhm_vs_z_metrics:
        metrics["residual_fwhm_vs_true_z"] = fwhm_vs_z_metrics

    energy_trend_metrics = plot_residual_trends_vs_energy_proxy(
        y_true,
        y_pred,
        metadata,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        n_bins=n_z_resolution_bins,
    )
    if energy_trend_metrics:
        metrics["residual_trends_vs_energy_proxy"] = energy_trend_metrics

    process_z_trend_metrics = plot_residual_trends_vs_true_z_by_process(
        y_true,
        y_pred,
        metadata,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        process_column=process_column,
        n_z_bins=n_z_resolution_bins,
    )
    if process_z_trend_metrics:
        metrics["residual_trends_vs_true_z_by_process"] = process_z_trend_metrics

    coordinate_distribution_metrics = plot_true_predicted_coordinate_distributions(
        y_true,
        y_pred,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        predicted_uncertainties=predicted_uncertainties,
    )
    if coordinate_distribution_metrics:
        metrics["true_vs_predicted_coordinate_distributions"] = coordinate_distribution_metrics

    true_z_by_pred_z_metrics = plot_true_z_by_predicted_z_bins(
        y_true,
        y_pred,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        n_pred_z_bins=n_pred_z_condition_bins,
        gmm_components=n_pred_z_condition_gmm_components,
    )
    if true_z_by_pred_z_metrics:
        metrics["true_z_by_predicted_z_bins"] = true_z_by_pred_z_metrics

    doi_observable_metrics = plot_doi_observables_vs_true_z(
        y_true,
        metadata,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
    )
    if doi_observable_metrics:
        metrics["doi_observables_vs_true_z"] = doi_observable_metrics

    pe_spectrum_metrics = plot_total_pe_energy_spectrum(
        metadata,
        outdir,
        width_text_label=r"Gaussian FWHM$/\mu \times 100$",
        mode_min=pe_spectrum_mode_min,
        fit_fraction=pe_spectrum_fit_fraction,
        fit_low_fraction=pe_spectrum_fit_low_fraction,
        fit_high_fraction=pe_spectrum_fit_high_fraction,
        fit_model=pe_spectrum_fit_model,
    )
    if pe_spectrum_metrics:
        metrics["total_pe_energy_spectrum"] = pe_spectrum_metrics
        mode = pe_spectrum_metrics.get("mode", np.nan)
        if np.isfinite(mode) and mode > 0:
            total_visible_energy_metrics = plot_total_pe_energy_spectrum(
                metadata,
                outdir,
                output_filename="total-visible-energy-spectrum.png",
                x_label=r"$E^{\mathrm{total}}_{\mathrm{vis}}$ (keV)",
                width_text_label=r"Gaussian FWHM$/\mu \times 100$",
                mode_min=pe_spectrum_mode_min,
                fit_fraction=pe_spectrum_fit_fraction,
                fit_low_fraction=pe_spectrum_fit_low_fraction,
                fit_high_fraction=pe_spectrum_fit_high_fraction,
                fit_model=pe_spectrum_fit_model,
                x_scale=VISIBLE_ENERGY_CALIBRATION_KEV / mode,
                spectrum_feature_labels=_default_total_visible_energy_feature_labels(),
            )
            if total_visible_energy_metrics:
                metrics["total_visible_energy_spectrum"] = total_visible_energy_metrics

    max_pe_spectrum_metrics = plot_total_pe_energy_spectrum(
        metadata,
        outdir,
        pe_column="max_pe",
        output_filename="max-pixel-pe-spectrum.png",
        x_label=r"$N^{\mathrm{max}}_{\mathrm{PE}}$",
        width_text_label=r"$\mathrm{FWHM}/\mu$",
        show_fit=False,
        annotate_max_pe_features=True,
        max_pe_feature_positions=max_pe_feature_positions,
        mode_min=pe_spectrum_mode_min,
        fit_fraction=pe_spectrum_fit_fraction,
        fit_low_fraction=pe_spectrum_fit_low_fraction,
        fit_high_fraction=pe_spectrum_fit_high_fraction,
        fit_model=pe_spectrum_fit_model,
    )
    if max_pe_spectrum_metrics:
        metrics["max_pixel_pe_spectrum"] = max_pe_spectrum_metrics

    gamma_depth_metrics = plot_gamma_first_interaction_depth_validation(
        y_true,
        metadata,
        outdir,
        output_names,
        spatial_indices=spatial_indices,
        pixel_geometry=pixel_geometry,
        incident_energy_window_mev=incident_energy_window_mev,
        source_radius_mm=source_radius_mm,
        source_distance_cm=source_distance_cm,
        cone_margin=cone_margin,
        cone_half_angle_deg=cone_half_angle_deg,
        cone_angle_samples=cone_angle_samples,
    )
    if gamma_depth_metrics:
        metrics["gamma_first_interaction_depth_validation"] = gamma_depth_metrics

    incident_angle_metrics = plot_gamma_incident_angle_distribution(
        metadata,
        outdir,
    )
    if incident_angle_metrics:
        metrics["gamma_incident_angle_distribution"] = incident_angle_metrics

    if pixel_geometry is not None:
        pixel_region_residual_metrics = plot_residuals_by_pixel_region(
            y_true,
            y_pred,
            metadata,
            outdir,
            output_names,
            pixel_geometry,
            spatial_indices=spatial_indices,
        )
        if pixel_region_residual_metrics:
            metrics["residuals_by_pixel_region"] = pixel_region_residual_metrics

        pixel_region_trend_metrics = plot_residual_trends_vs_true_z_by_pixel_region(
            y_true,
            y_pred,
            metadata,
            outdir,
            output_names,
            pixel_geometry,
            spatial_indices=spatial_indices,
            n_z_bins=n_z_resolution_bins,
        )
        if pixel_region_trend_metrics:
            metrics["residual_trends_vs_true_z_by_pixel_region"] = pixel_region_trend_metrics

    if pixel_geometry is not None and make_per_pixel_plots:
        z_index = spatial_indices[2]
        metrics["per_pixel"] = plot_per_pixel_doi_diagnostics(
            y_true,
            y_pred,
            metadata,
            outdir,
            z_index,
            pixel_geometry,
            min_events=min_events_per_pixel,
        )
    elif pixel_geometry is not None:
        metrics["per_pixel"] = {"skipped": True, "reason": "disabled by --no-per-pixel-plots"}

    report_path = write_html_report(outdir, metrics)
    metrics["html_report"] = str(report_path)

    return metrics


def _parse_standalone_args():
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate the current CNN/GMM performance and validation plots "
            "from a predictions CSV."
        )
    )
    parser.add_argument("--predictions-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--incident-energy-window-kev", type=float, default=1.0)
    parser.add_argument("--pe-spectrum-mode-min", type=float, default=PE_SPECTRUM_MODE_MIN)
    parser.add_argument("--pe-spectrum-fit-fraction", type=float, default=PE_SPECTRUM_FIT_FRACTION)
    parser.add_argument("--pe-spectrum-fit-low-fraction", type=float, default=None)
    parser.add_argument("--pe-spectrum-fit-high-fraction", type=float, default=None)
    parser.add_argument("--max-pe-compton-edge-x", type=float, default=None)
    parser.add_argument("--max-pe-klein-nishina-maximum-x", type=float, default=None)
    parser.add_argument("--max-pe-photopeak-x", type=float, default=None)
    parser.add_argument(
        "--no-per-pixel-plots",
        action="store_true",
        help="Skip per-pixel residual plots and DOI resolution map.",
    )
    parser.add_argument("--n-z-resolution-bins", type=int, default=8)
    parser.add_argument("--n-pred-z-condition-bins", type=int, default=8)
    parser.add_argument(
        "--n-pred-z-condition-gmm-components",
        type=int,
        default=N_PRED_Z_CONDITION_GMM_COMPONENTS,
        help="Number of GMM components fitted to true-z distributions in predicted-z bins.",
    )
    parser.add_argument(
        "--z-gmm-point-estimators",
        nargs="+",
        choices=("mean", "median", "mode", "input"),
        default=("mean",),
        help="GMM-derived z point estimators to evaluate.",
    )
    return parser.parse_args()

def _is_sipm_pixel_pe_column(name, image_size=8):
    if not name.startswith("pe_"):
        return False
    parts = name.split("_")
    if len(parts) != 3:
        return False
    try:
        ix = int(parts[1])
        iy = int(parts[2])
    except ValueError:
        return False
    return 0 <= ix < image_size and 0 <= iy < image_size


def _standalone_usecols(
    header_cols,
    output_names,
    image_size,
    include_predictions,
    include_sipm_pixel_columns,
):
    """Read only columns used by the requested standalone evaluation mode."""
    header_set = set(header_cols)
    true_cols = _infer_true_target_columns_from_columns(header_cols, output_names)
    if true_cols is None:
        return None

    usecols = set(true_cols)

    validation_columns = {
        "total_pe",
        "max_pe",
        "frac_max",
        "spread",
        "z_gamma",
        "gamma_incident_energy",
        "gamma_incident_angle_deg",
    }
    usecols.update(col for col in validation_columns if col in header_set)

    if include_predictions:
        prediction_columns = {
            "process",
            "gamma_delta_ke",
            "x_gamma",
            "y_gamma",
            "pe_3x3_around_max",
            "frac_3x3_around_max",
            "z_gmm_mean",
        }
        prediction_columns.update(f"{name}_pred" for name in output_names)
        prediction_columns.update(f"{name}_sigma" for name in output_names)
        usecols.update(col for col in prediction_columns if col in header_set)

        # Keep GMM component columns if present; these are needed only for old
        # prediction CSVs that do not already contain z_gmm_mean.
        usecols.update(col for col in header_cols if col.startswith("z_gmm_"))

    if include_sipm_pixel_columns:
        usecols.update(
            col for col in header_cols
            if _is_sipm_pixel_pe_column(col, image_size=image_size)
        )

    return [col for col in header_cols if col in usecols]


def _read_standalone_csv(args):
    header_cols = list(pd.read_csv(args.predictions_csv, nrows=0, memory_map=True).columns)
    pred_cols = [f"{name}_pred" for name in OUTPUT_NAMES]
    include_predictions = all(col in header_cols for col in pred_cols)
    usecols = _standalone_usecols(
        header_cols,
        OUTPUT_NAMES,
        image_size=PIXEL_GEOMETRY["image_size"],
        include_predictions=include_predictions,
        include_sipm_pixel_columns=include_predictions and not args.no_per_pixel_plots,
    )

    if usecols is None:
        raise RuntimeError("Missing required true coordinate columns: x/y/z_true or x/y/z_gamma")

    print(f"Reading {len(usecols)} / {len(header_cols)} CSV columns")
    return pd.read_csv(args.predictions_csv, usecols=usecols, memory_map=True)


def _run_standalone():
    args = _parse_standalone_args()
    df = _read_standalone_csv(args)

    true_cols = _infer_true_target_columns(df, OUTPUT_NAMES)
    if true_cols is None:
        raise RuntimeError("Missing required true coordinate columns: x/y/z_true or x/y/z_gamma")

    y_true = df[true_cols].to_numpy(dtype=float)
    pred_cols = [f"{name}_pred" for name in OUTPUT_NAMES]
    sigma_cols = [f"{name}_sigma" for name in OUTPUT_NAMES]
    max_pe_feature_positions = {
        "compton_edge": args.max_pe_compton_edge_x,
        "klein_nishina_maximum": args.max_pe_klein_nishina_maximum_x,
        "xray_escape": None,
        "photopeak": args.max_pe_photopeak_x,
    }

    if not all(c in df.columns for c in pred_cols):
        print("Prediction columns were not found; running validation-only plots.")
        metrics = evaluate_validation_plots(
            y_true=y_true,
            outdir=args.outdir,
            output_names=OUTPUT_NAMES,
            spatial_indices=SPATIAL_INDICES,
            metadata=df,
            pixel_geometry=PIXEL_GEOMETRY,
            incident_energy_window_mev=args.incident_energy_window_kev / 1000.0,
            source_radius_mm=SOURCE_RADIUS_MM,
            source_distance_cm=SOURCE_DISTANCE_CM,
            cone_margin=CONE_MARGIN,
            cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
            cone_angle_samples=CONE_ANGLE_SAMPLES,
            pe_spectrum_mode_min=args.pe_spectrum_mode_min,
            pe_spectrum_fit_fraction=args.pe_spectrum_fit_fraction,
            pe_spectrum_fit_low_fraction=args.pe_spectrum_fit_low_fraction,
            pe_spectrum_fit_high_fraction=args.pe_spectrum_fit_high_fraction,
            pe_spectrum_fit_model=PE_SPECTRUM_FIT_MODEL,
            max_pe_feature_positions=max_pe_feature_positions,
        )
        args.outdir.mkdir(parents=True, exist_ok=True)
        with open(args.outdir / "metrics-standalone-eval.json", "w") as f:
            json.dump(metrics, f, indent=2)
        return

    y_pred = df[pred_cols].to_numpy(dtype=float)
    predicted_uncertainties = None
    if all(c in df.columns for c in sigma_cols):
        predicted_uncertainties = df[sigma_cols].to_numpy(dtype=float)

    all_metrics = {}
    estimators = tuple(args.z_gmm_point_estimators)
    for estimator in estimators:
        estimator_outdir = args.outdir if len(estimators) == 1 else args.outdir / f"z-gmm-{estimator}"
        metrics = evaluate_model_performance(
            y_true=y_true,
            y_pred=y_pred,
            outdir=estimator_outdir,
            output_names=OUTPUT_NAMES,
            spatial_indices=SPATIAL_INDICES,
            predicted_uncertainties=predicted_uncertainties,
            metadata=df,
            process_column="process",
            pixel_geometry=PIXEL_GEOMETRY,
            min_events_per_pixel=MIN_EVENTS_PER_PIXEL,
            n_z_resolution_bins=args.n_z_resolution_bins,
            incident_energy_window_mev=args.incident_energy_window_kev / 1000.0,
            source_radius_mm=SOURCE_RADIUS_MM,
            source_distance_cm=SOURCE_DISTANCE_CM,
            cone_margin=CONE_MARGIN,
            cone_half_angle_deg=CONE_HALF_ANGLE_DEG,
            cone_angle_samples=CONE_ANGLE_SAMPLES,
            pe_spectrum_mode_min=args.pe_spectrum_mode_min,
            pe_spectrum_fit_fraction=args.pe_spectrum_fit_fraction,
            pe_spectrum_fit_low_fraction=args.pe_spectrum_fit_low_fraction,
            pe_spectrum_fit_high_fraction=args.pe_spectrum_fit_high_fraction,
            pe_spectrum_fit_model=PE_SPECTRUM_FIT_MODEL,
            max_pe_feature_positions=max_pe_feature_positions,
            make_per_pixel_plots=not args.no_per_pixel_plots,
            n_pred_z_condition_bins=args.n_pred_z_condition_bins,
            n_pred_z_condition_gmm_components=args.n_pred_z_condition_gmm_components,
            z_gmm_point_estimator=estimator,
            z_gmm_mode_grid_size=Z_GMM_MODE_GRID_SIZE,
        )
        estimator_outdir.mkdir(parents=True, exist_ok=True)
        with open(estimator_outdir / "metrics-standalone-eval.json", "w") as f:
            json.dump(metrics, f, indent=2)
        all_metrics[estimator] = {
            "outdir": str(estimator_outdir),
            "z_point_prediction": metrics.get("z_point_prediction"),
            "mean_3d_error_mm": metrics.get("mean_3d_error_mm"),
            "p68_3d_error_mm": metrics.get("p68_3d_error_mm"),
            "p95_3d_error_mm": metrics.get("p95_3d_error_mm"),
        }

    if len(estimators) > 1:
        args.outdir.mkdir(parents=True, exist_ok=True)
        with open(args.outdir / "metrics-z-gmm-point-estimator-comparison.json", "w") as f:
            json.dump(all_metrics, f, indent=2)
        write_html_report(args.outdir, {"z_gmm_point_estimator_comparison": all_metrics})


if __name__ == "__main__":
    _run_standalone()
