#!/usr/bin/env python3
"""Plot CNN performance metrics versus the number of z-GMM components.

The script expects evaluation directories named like ``cnn-gmm-n16-50M``.
Each directory should contain per-estimator metric files such as:

    z-gmm-mean/metrics-standalone-eval.json
    z-gmm-median/metrics-standalone-eval.json
    z-gmm-mode/metrics-standalone-eval.json

The plots compare all events and the subset passing the per-axis requirement
``predicted uncertainty < median(predicted uncertainty)``.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


AXES = ("x", "y", "z")
ESTIMATORS = ("mean", "median", "mode")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot performance versus z-GMM component count."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("."),
        help="Directory containing cnn-gmm-n*-50M evaluation folders.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("model-comparison"),
        help="Directory for comparison plots and summary CSV.",
    )
    parser.add_argument(
        "--pattern",
        default="cnn-gmm-n*-50M",
        help="Glob pattern for model evaluation directories.",
    )
    return parser.parse_args()


def component_count(path: Path) -> int | None:
    match = re.search(r"cnn-gmm-n(\d+)", path.name)
    return int(match.group(1)) if match else None


def metrics_path(model_dir: Path, estimator: str) -> Path:
    return model_dir / f"z-gmm-{estimator}" / "metrics-standalone-eval.json"


def nested_get(mapping: dict, keys, default=None):
    value = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def read_rows(input_dir: Path, pattern: str) -> pd.DataFrame:
    rows = []
    for model_dir in sorted(input_dir.glob(pattern), key=lambda p: component_count(p) or -1):
        n_components = component_count(model_dir)
        if n_components is None:
            continue

        for estimator in ESTIMATORS:
            path = metrics_path(model_dir, estimator)
            if not path.exists():
                continue

            with path.open() as f:
                metrics = json.load(f)

            for axis in AXES:
                pull_metrics = metrics.get("pull", {}).get(axis, {})
                cut_metrics = metrics.get("residuals_with_uncertainty_cut", {}).get(axis, {})
                rows.append(
                    {
                        "n_components": n_components,
                        "estimator": estimator,
                        "axis": axis,
                        "fit_fwhm_mm": pull_metrics.get("fit_fwhm"),
                        "fit_sigma_mm": pull_metrics.get("fit_sigma"),
                        "fit_mu_mm": pull_metrics.get("fit_mu"),
                        "mean_abs_error_mm": metrics.get("spatial_mae_mm", {}).get(axis),
                        "population_std_mm": nested_get(cut_metrics, ("all", "std"), pull_metrics.get("resid_std")),
                        "low_uncertainty_rmse_mm": nested_get(
                            cut_metrics, ("sigma_cut", "rmse")
                        ),
                        "low_uncertainty_fit_fwhm_mm": nested_get(
                            cut_metrics, ("sigma_cut_gaussian_fit", "fwhm")
                        ),
                        "low_uncertainty_population_std_mm": nested_get(
                            cut_metrics, ("sigma_cut", "std")
                        ),
                        "low_uncertainty_selected_fraction": cut_metrics.get("selected_fraction"),
                        "fit_ok": pull_metrics.get("fit_ok"),
                        "low_uncertainty_fit_ok": nested_get(
                            cut_metrics, ("sigma_cut_gaussian_fit", "fit_ok")
                        ),
                        "mean_3d_error_mm": metrics.get("mean_3d_error_mm"),
                        "median_3d_error_mm": metrics.get("median_3d_error_mm"),
                        "p68_3d_error_mm": metrics.get("p68_3d_error_mm"),
                    }
                )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No metric files found in {input_dir} matching {pattern}")
    return df.sort_values(["axis", "estimator", "n_components"])


def plot_metric_vs_components(
    df: pd.DataFrame,
    outdir: Path,
    metric_column: str,
    ylabel_template: str,
    output_filename: str,
    fit_ok_column: str | None = None,
):
    colors = {
        "mean": "tab:blue",
        "median": "tab:orange",
        "mode": "tab:green",
    }
    markers = {
        "mean": "o",
        "median": "s",
        "mode": "^",
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True)
    for ax, axis in zip(axes, AXES):
        axis_df = df[df["axis"] == axis].copy()
        axis_df[metric_column] = pd.to_numeric(axis_df[metric_column], errors="coerce")
        axis_df = axis_df[axis_df[metric_column].notna()]
        if fit_ok_column is not None and fit_ok_column in axis_df.columns:
            axis_df = axis_df[axis_df[fit_ok_column] == True]

        for estimator in ESTIMATORS:
            plot_df = axis_df[axis_df["estimator"] == estimator]
            if plot_df.empty:
                continue
            ax.plot(
                plot_df["n_components"],
                plot_df[metric_column],
                marker=markers[estimator],
                color=colors[estimator],
                linewidth=1.8,
                label=estimator,
            )

        ax.set_xlabel("Number of z-GMM components", loc="right")
        ax.set_ylabel(ylabel_template.format(axis=axis), loc="top")
        ax.tick_params(labelsize=11)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(frameon=False, fontsize=11)
    fig.tight_layout()
    fig.savefig(outdir / output_filename, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    df = read_rows(args.input_dir, args.pattern)
    df.to_csv(args.outdir / "gmm-component-performance-summary.csv", index=False)
    plot_metric_vs_components(
        df,
        args.outdir,
        metric_column="fit_fwhm_mm",
        ylabel_template="{axis} residual fitted FWHM (mm)",
        output_filename="residual-fwhm-vs-gmm-components.png",
        fit_ok_column="fit_ok",
    )
    plot_metric_vs_components(
        df,
        args.outdir,
        metric_column="mean_abs_error_mm",
        ylabel_template="{axis} mean absolute residual (mm)",
        output_filename="mean-absolute-residual-vs-gmm-components.png",
    )
    plot_metric_vs_components(
        df,
        args.outdir,
        metric_column="population_std_mm",
        ylabel_template="{axis} residual population std. dev. (mm)",
        output_filename="residual-population-std-vs-gmm-components.png",
    )
    plot_metric_vs_components(
        df,
        args.outdir,
        metric_column="low_uncertainty_fit_fwhm_mm",
        ylabel_template="{axis} residual fitted FWHM (mm)\n$\\sigma_{{\\mathrm{{pred}}}} < \\mathrm{{median}}$",
        output_filename="low-uncertainty-residual-fwhm-vs-gmm-components.png",
        fit_ok_column="low_uncertainty_fit_ok",
    )
    plot_metric_vs_components(
        df,
        args.outdir,
        metric_column="low_uncertainty_rmse_mm",
        ylabel_template="{axis} residual RMSE (mm)\n$\\sigma_{{\\mathrm{{pred}}}} < \\mathrm{{median}}$",
        output_filename="low-uncertainty-residual-rmse-vs-gmm-components.png",
    )
    plot_metric_vs_components(
        df,
        args.outdir,
        metric_column="low_uncertainty_population_std_mm",
        ylabel_template="{axis} residual population std. dev. (mm)\n$\\sigma_{{\\mathrm{{pred}}}} < \\mathrm{{median}}$",
        output_filename="low-uncertainty-residual-population-std-vs-gmm-components.png",
    )

    print(f"Wrote {args.outdir / 'gmm-component-performance-summary.csv'}")
    print(f"Wrote plots to {args.outdir}")


if __name__ == "__main__":
    main()
