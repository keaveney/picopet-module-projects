# analyse.py
# this file:
# reads in root files with output of gamma scintillation and optical transport simulation
# assigns each optical photon to a virtual sipm pixel, applies a PDE, and estimates n photoelectrons in each sipm pixel
# optionally applies a total-PE window cut
# makes features dataframe: high-level features, per-pixel n photoelectrons
# adds truth-level interaction coordinates from separate output file by matching with Event ID
# write resulting dataframe to disk

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import uproot

# Geometry
pix = 3.0   # mm
foil = 0.2  # mm
n = 8
pitch = pix + foil
offset = (n - 1) * pitch / 2.0

# Defaults
PDE = 0.30
USE_STREAMING = True
STEP_SIZE = 500_000
ENERGY_WINDOW = None                 # default: write all events; set bounds with CLI args if needed
MAX_PE_FIT_WINDOW = (700.0, 1600.0)  # fit window for max_pe histogram; set to None to fit full range
PE_MODE_MIN = 100.0                  # ignore low-PE pileup when finding spectrum mode

def optional_float(value):
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in {"none", "null", "nan"}:
        return None
    return float(value)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sipm-root", type=str, required=True, help="Path to sipm_hits.root")
    parser.add_argument("--gamma-root", type=str, required=True, help="Path to gamma_steps.root")
    parser.add_argument("--output-csv", type=str, required=True, help="Path to output CSV")
    parser.add_argument("--plot-dir", type=str, required=True, help="Directory for analysis plots")
    parser.add_argument("--rng-seed", type=int, default=123, help="Seed for PDE binomial sampling")
    parser.add_argument("--step-size", type=int, default=STEP_SIZE, help="Uproot iteration step size")
    parser.add_argument("--total-pe-min", type=optional_float, default=None, help="Optional lower total_pe cut; use None to disable")
    parser.add_argument("--total-pe-max", type=optional_float, default=None, help="Optional upper total_pe cut; use None to disable")
    return parser.parse_args()


def _find_tree_key(root_file):
    with uproot.open(root_file) as f:
        classnames = f.classnames()
        tree_key = next((k for k, v in classnames.items() if v.endswith("TTree")), None)
        if tree_key is None:
            raise RuntimeError(f"No TTree found in {root_file}. Keys: {list(classnames.keys())}")
        return tree_key


def _gauss_plus_falling_exp(x, A, mu, sigma, B, lamb, x0):
    gauss = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    expo = B * np.exp(-lamb * (x - x0))
    return gauss + expo


def plot_pe_spectrum_with_fit(values, xlabel, title, output_path, bins=50):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    values_for_mode = values[values > PE_MODE_MIN]

    plt.figure(figsize=(6, 4))
    counts, edges, _ = plt.hist(values, bins=bins, alpha=0.85)
    plt.xlabel(xlabel, loc="right", fontsize=10)
    plt.ylabel("Counts", loc="top", fontsize=10)
    #plt.title(title) #better without title

    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(values_for_mode) > 0 and len(centers) > 1 and np.any(counts > 0):
        mode_counts, mode_edges = np.histogram(values_for_mode, bins=edges)
        mode_centers = 0.5 * (mode_edges[:-1] + mode_edges[1:])
        mode = float(mode_centers[np.argmax(mode_counts)])

        # Start with the requested mode +/-35% window, but expand it if the
        # histogram is sparse. This avoids silently skipping the fit for
        # max-pixel or 3x3 spectra where the initial window can contain too few
        # populated bins for a 5-parameter model.
        min_populated_bins = 6
        fit_fraction = 0.35
        fit_mask = np.zeros_like(centers, dtype=bool)
        while fit_fraction <= 2.0:
            lo = max(0.0, (1.0 - fit_fraction) * mode)
            hi = (1.0 + fit_fraction) * mode
            candidate = (centers >= lo) & (centers <= hi) & (counts > 0)
            if np.count_nonzero(candidate) >= min_populated_bins:
                fit_mask = candidate
                break
            fit_fraction += 0.25

        if not np.any(fit_mask):
            fit_mask = (centers >= PE_MODE_MIN) & (counts > 0)

        fit_counts = counts[fit_mask]
        fit_centers = centers[fit_mask]
        fit_window = (float(fit_centers[0]), float(fit_centers[-1])) if len(fit_centers) else (np.nan, np.nan)

        if len(fit_centers) >= min_populated_bins and np.any(fit_counts > 0):
            x0 = float(fit_centers[0])
            A0 = float(fit_counts.max())
            mu0 = float(np.average(fit_centers, weights=fit_counts))
            sigma0 = max(0.1 * mode, 1.0)
            B0 = max(A0 * 0.2, 1.0)
            lamb0 = 1.0 / max(float(np.mean(fit_centers) - x0), 1.0)

            try:
                from scipy.optimize import curve_fit

                lower = [0.0, float(fit_centers[0]), 1e-6, 0.0, 0.0]
                upper = [np.inf, float(fit_centers[-1]), np.inf, np.inf, np.inf]
                popt, _ = curve_fit(
                    lambda x, A, mu, sigma, B, lamb: _gauss_plus_falling_exp(x, A, mu, sigma, B, lamb, x0),
                    fit_centers,
                    fit_counts,
                    p0=[A0, mu0, sigma0, B0, lamb0],
                    bounds=(lower, upper),
                    maxfev=20000,
                )
                A_fit, mu_fit, sigma_fit, B_fit, lamb_fit = popt
                sigma_fit = abs(float(sigma_fit))
                fwhm = 2.355 * sigma_fit
                fwhm_pct = 100.0 * fwhm / mu_fit if mu_fit != 0 else np.nan

                xfit = np.linspace(float(fit_centers[0]), float(fit_centers[-1]), 400)
                gauss_fit = A_fit * np.exp(-0.5 * ((xfit - mu_fit) / sigma_fit) ** 2)
                exp_fit = B_fit * np.exp(-lamb_fit * (xfit - x0))
                total_fit = gauss_fit + exp_fit

                #plt.axvspan(fit_window[0], fit_window[1], color="gray", alpha=0.12, label="Fit window")
                plt.plot(xfit, total_fit, "r-", linewidth=2, label="Gaussian + exponential")
                plt.plot(xfit, gauss_fit, "b--", linewidth=1.5, label="Gaussian component")
                plt.plot(xfit, exp_fit, "g--", linewidth=1.5, label="Exponential component")

                txt = (
                    #f"mode = {mode:.3g} (>{PE_MODE_MIN:.0f})\n"
                    #f"fit window = [{fit_window[0]:.3g}, {fit_window[1]:.3g}]\n"
                    #f"$\\mu$ = {mu_fit:.3g}\n"
                    #f"$\\sigma$ = {sigma_fit:.3g}\n"
                    #f"FWHM = {fwhm:.3g}\n"
                    f"(FWHM / $\\mu) \\times 100 $ = {fwhm_pct:.2f}%"
                )
                plt.text(
                    0.33,
                    0.76,
                    txt,
                    transform=plt.gca().transAxes,
                    ha="right",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white"),
                    fontsize=10,
                )
                plt.legend(fontsize=10, loc="upper left")
            except Exception as exc:
                print(f"Could not fit {output_path.name}: {exc}")
        else:
            print(
                f"Could not fit {output_path.name}: only {len(fit_centers)} populated bins "
                f"after dynamic fit-window selection"
            )

    #plt.tight_layout(pad=0.2)
    plt.subplots_adjust(left=-0.1, right=0.95, bottom=0.13, top=0.99)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close()


def _fit_truncated_exponential_mu(depth_mm, length_mm):
    depth_mm = np.asarray(depth_mm, dtype=float)
    depth_mm = depth_mm[np.isfinite(depth_mm)]
    if len(depth_mm) < 5 or length_mm <= 0:
        return {"fit_ok": False, "mu_per_mm": np.nan, "attenuation_length_mm": np.nan}

    mean_depth = float(np.mean(depth_mm))
    if mean_depth >= 0.5 * length_mm:
        return {"fit_ok": False, "mu_per_mm": 0.0, "attenuation_length_mm": np.inf}

    def score(mu):
        mu_l = mu * length_mm
        if mu_l < 1e-6:
            expected_mean = 0.5 * length_mm - (mu * length_mm**2) / 12.0
            return expected_mean - mean_depth
        return 1.0 / mu - length_mm / np.expm1(mu_l) - mean_depth

    lo = 1e-12
    hi = 1.0 / length_mm
    while score(hi) > 0 and hi < 1e3:
        hi *= 2.0

    if score(hi) > 0:
        mu = hi
        fit_ok = False
    else:
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if score(mid) > 0:
                lo = mid
            else:
                hi = mid
        mu = 0.5 * (lo + hi)
        fit_ok = True

    return {
        "fit_ok": bool(fit_ok),
        "mu_per_mm": float(mu),
        "attenuation_length_mm": float(1.0 / mu) if mu > 0 else np.inf,
    }


def plot_raw_gamma_depth_validation(gdf_primary, plot_dir, pixel_length_mm=15.0, incident_energy_window_mev=0.001):
    """Plot first gamma interaction depth before SiPM matching or total_pe cuts."""
    if len(gdf_primary) == 0:
        print("Skipping raw gamma depth validation: no primary gamma interactions")
        return

    z_front = -0.5 * pixel_length_mm
    depth = gdf_primary["z_gamma"].to_numpy(dtype=float) - z_front
    valid = np.isfinite(depth) & (depth >= 0.0) & (depth <= pixel_length_mm)

    if "incident_energy" in gdf_primary.columns:
        energy = gdf_primary["incident_energy"].to_numpy(dtype=float)
        finite_energy = np.isfinite(energy)
        if np.any(finite_energy):
            energy_mev = energy / 1000.0 if np.nanmedian(energy[finite_energy]) > 10.0 else energy
            energy_mask = finite_energy & (np.abs(energy_mev - 0.511) <= incident_energy_window_mev)
            if np.sum(valid & energy_mask) >= 50:
                valid &= energy_mask

    depth = depth[valid]
    depth = depth[np.isfinite(depth)]
    if len(depth) < 50:
        print(f"Skipping raw gamma depth validation: only {len(depth)} selected interactions")
        return

    bins = np.linspace(0.0, pixel_length_mm, 41)
    counts, edges = np.histogram(depth, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fit = _fit_truncated_exponential_mu(depth, pixel_length_mm)

    if fit["mu_per_mm"] > 0:
        mu = fit["mu_per_mm"]
        norm = 1.0 - np.exp(-mu * pixel_length_mm)
        expected = len(depth) * (np.exp(-mu * edges[:-1]) - np.exp(-mu * edges[1:])) / max(norm, 1e-12)
    else:
        expected = len(depth) * np.diff(edges) / pixel_length_mm
    expected = expected / np.sum(expected) if np.sum(expected) > 0 else expected

    plt.figure(figsize=(6.4, 4.4))
    plt.hist(
        depth,
        bins=bins,
        weights=np.full(len(depth), 1.0 / len(depth)),
        histtype="stepfilled",
        alpha=0.55,
        color="tab:blue",
        label="simulation",
    )
    plt.step(
        centers,
        expected,
        where="mid",
        color="tab:red",
        linewidth=2.0,
        label="attenuation model",
    )
    plt.xlabel("First interaction z (mm)", loc="right")
    plt.ylabel("Fraction of events / bin", loc="top")
    attenuation = fit["attenuation_length_mm"]
    txt = rf"$\lambda = {attenuation:.3g}\,\mathrm{{mm}}$" if np.isfinite(attenuation) else r"$\lambda \to \infty$"
    plt.text(
        0.97,
        0.94,
        txt,
        transform=plt.gca().transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82),
        fontsize=10,
    )
    plt.legend(fontsize=10, loc="best")
    plt.tight_layout()
    plt.savefig(Path(plot_dir) / "raw-gamma-first-interaction-depth-attenuation.png")
    plt.close()


def build_npe_df(filename, rng_seed, step_size=500_000):
    root_file = Path(filename)

    if not root_file.exists():
        roots = sorted(root_file.parent.glob("*.root"))
        raise FileNotFoundError(f"{root_file} not found. Available: {[p.name for p in roots]}")

    tree_key = _find_tree_key(root_file)

    # Determine how positions are stored to keep IO minimal.
    with uproot.open(root_file) as f:
        tree = f[tree_key]
        branches = set(tree.keys())

    has_xyz = {"Position_X", "Position_Y"}.issubset(branches)
    has_vec = "Position" in branches

    if has_xyz:
        filter_name = ["EventID", "Position_X", "Position_Y"]
        pos_mode = "xyz"
    elif has_vec:
        filter_name = ["EventID", "Position"]
        pos_mode = "vec"
    else:
        raise RuntimeError("Position branches not found (expected Position_X/Position_Y or Position).")

    counts = {}

    for arrays in uproot.iterate(f"{root_file}:{tree_key}", filter_name=filter_name, step_size=step_size, library="np"):
        event = arrays["EventID"]
        if pos_mode == "xyz":
            x_mm = arrays["Position_X"]
            y_mm = arrays["Position_Y"]
        else:
            pos = arrays["Position"]
            x_mm = pos[:, 0]
            y_mm = pos[:, 1]

        ix = np.round((x_mm + offset) / pitch).astype(np.int16)
        iy = np.round((y_mm + offset) / pitch).astype(np.int16)

        cx = ix * pitch - offset
        cy = iy * pitch - offset

        in_bounds = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        inside_pixel = in_bounds & (np.abs(x_mm - cx) <= pix / 2.0) & (np.abs(y_mm - cy) <= pix / 2.0)

        if not np.any(inside_pixel):
            continue

        ev = event[inside_pixel]
        ixs = ix[inside_pixel]
        iys = iy[inside_pixel]

        keys = np.stack([ev, ixs, iys], axis=1)
        uniq, cnts = np.unique(keys, axis=0, return_counts=True)
        for (evt, ixv, iyv), c in zip(uniq, cnts):
            key = (int(evt), int(ixv), int(iyv))
            counts[key] = counts.get(key, 0) + int(c)

    rows = [(evt, ixv, iyv, n_photons) for (evt, ixv, iyv), n_photons in counts.items()]
    df_counts = pd.DataFrame(rows, columns=["event", "ix", "iy", "n_photons"])

    rng = np.random.default_rng(rng_seed)
    df_counts["n_pe"] = rng.binomial(df_counts["n_photons"].to_numpy(), PDE)

    return df_counts


def build_doi_df(df_npe_pixels, pe_window, plot_dir):
    features = []

    for evt, grp in df_npe_pixels.groupby("event"):
        total_pe = grp["n_pe"].sum()
        if total_pe == 0:
            continue

        max_pe = grp["n_pe"].max()
        frac_max = max_pe / total_pe

        xs = grp["ix"].to_numpy()
        ys = grp["iy"].to_numpy()
        w = grp["n_pe"].to_numpy()
        max_row = grp.loc[grp["n_pe"].idxmax()]
        max_ix = int(max_row["ix"])
        max_iy = int(max_row["iy"])
        in_3x3 = (np.abs(xs - max_ix) <= 1) & (np.abs(ys - max_iy) <= 1)
        pe_3x3_max = int(w[in_3x3].sum())

        cx = np.average(xs, weights=w)
        cy = np.average(ys, weights=w)
        spread = np.sqrt(np.average((xs - cx) ** 2 + (ys - cy) ** 2, weights=w))

        features.append(
            {
                "event": evt,
                "total_pe": total_pe,
                "max_pe": max_pe,
                "pe_3x3_max": pe_3x3_max,
                "frac_max": frac_max,
                "spread": spread,
            }
        )

    df_feat = pd.DataFrame(features)

    pe_wide = df_npe_pixels.pivot_table(
        index="event", columns=["ix", "iy"], values="n_pe", aggfunc="sum", fill_value=0
    )
    pe_wide.columns = [f"pe_{ix}_{iy}" for ix, iy in pe_wide.columns]
    pe_wide = pe_wide.reset_index()
    df_feat = df_feat.merge(pe_wide, on="event", how="left")

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_pe_spectrum_with_fit(
        df_feat["total_pe"],
        "Total PE",
        "Total PE distribution",
        plot_dir / "total-pe-hist-nocuts.png",
        bins=50,
    )

    plot_pe_spectrum_with_fit(
        df_feat["pe_3x3_max"],
        "PE in 3x3 pixels centered on max pixel",
        "3x3 max-centered PE distribution",
        plot_dir / "max-centered-3x3-pe-hist-nocuts.png",
        bins=50,
    )

    plot_pe_spectrum_with_fit(
        df_feat["max_pe"],
        "NPE in Max pixel",
        "Max pixel PE distribution",
        plot_dir / "max-pixel-energy-nocuts.png",
        bins=40,
    )

    #this cut now looks a bit questionable - better to leave SIMs in and let CNN deduce (x,y,z)
    #hence we should apply a cut on total Npe?
    if pe_window is not None:
        lo, hi = pe_window
        if lo is not None:
            df_feat = df_feat[df_feat["total_pe"] >= lo]
        if hi is not None:
            df_feat = df_feat[df_feat["total_pe"] <= hi]

    return df_feat


def build_truth_df(df_doi, filename,  plot_dir):
    gamma_file = Path(filename)

    if not gamma_file.exists():
        raise FileNotFoundError(f"{gamma_file} not found")

    with uproot.open(gamma_file) as f:
        classnames = f.classnames()
        tree_key = next((k for k, v in classnames.items() if v.endswith("TTree")), None)
        if tree_key is None:
            raise RuntimeError(f"No TTree found in {gamma_file}. Keys: {list(classnames.keys())}")
        tree = f[tree_key]
        g = tree.arrays(library="np")

    def _gget(name):
        return g[name] if name in g else None

    def _gget_vec(base):
        arr = _gget(base)
        if arr is not None:
            return arr
        x = _gget(f"{base}_X")
        y = _gget(f"{base}_Y")
        z = _gget(f"{base}_Z")
        if x is None or y is None or z is None:
            return None
        return np.stack([x, y, z], axis=1)

    g_event = _gget("EventID")
    g_parent = _gget("ParentID")
    g_particle = _gget("ParticleName")
    g_process = _gget("ProcessDefinedStep")
    g_trackprocess = _gget("TrackCreatorProcess")
    g_pre_pos = _gget_vec("PrePosition")
    g_post_pos = _gget_vec("PostPosition")
    g_pos = g_post_pos
    if g_pos is None:
        g_pos = _gget_vec("Position")
    if g_pos is None:
        g_pos = g_pre_pos
    g_edep = _gget("TotalEnergyDeposit")
    g_pre_ke = _gget("PreKineticEnergy")
    g_post_ke = _gget("PostKineticEnergy")
    g_ke = _gget("KineticEnergy")
    g_time = _gget("GlobalTime")
    if g_event is None or g_pos is None or g_edep is None:
        raise RuntimeError("gamma_steps.root missing EventID, TotalEnergyDeposit, or Position/PrePosition/PostPosition")

    if g_pre_ke is not None:
        g_incident_energy = g_pre_ke
    elif g_ke is not None:
        g_incident_energy = g_ke
    else:
        g_incident_energy = np.full(len(g_event), np.nan)

    if g_pre_ke is not None and g_post_ke is not None:
        g_delta_ke = g_pre_ke - g_post_ke
    else:
        g_delta_ke = np.full(len(g_event), np.nan)

    if g_pre_pos is not None and g_post_pos is not None:
        step_vec = g_post_pos - g_pre_pos
        step_len = np.linalg.norm(step_vec, axis=1)
        cos_to_pixel_normal = np.full(len(g_event), np.nan)
        nonzero = step_len > 0
        # Use |cos(theta)| so the angle is with respect to the scintillator
        # pixel axis, independent of which side of the array the photon enters.
        cos_to_pixel_normal[nonzero] = np.abs(step_vec[nonzero, 2] / step_len[nonzero])
        g_incident_angle_deg = np.degrees(
            np.arccos(np.clip(cos_to_pixel_normal, 0.0, 1.0))
        )
    else:
        g_incident_angle_deg = np.full(len(g_event), np.nan)

    gdf = pd.DataFrame(
        {
            "event": g_event,
            "parent": g_parent if g_parent is not None else np.nan,
            "particle": g_particle.astype(str) if g_particle is not None else "",
            "process": g_process.astype(str) if g_process is not None else "",
            "trackprocess": g_trackprocess.astype(str) if g_trackprocess is not None else "",
            "edep": g_edep,
            "delta_ke": g_delta_ke,
            "incident_energy": g_incident_energy,
            "incident_angle_deg": g_incident_angle_deg,
            "time": g_time if g_time is not None else np.nan,
            "x_gamma": g_pos[:, 0],
            "y_gamma": g_pos[:, 1],
            "z_gamma": g_pos[:, 2],
        }
    )

    # Use the first primary gamma step with real energy deposit in the pixel as truth.
    # UserSpecialCut rows are technical terminations, not physical interactions.
    # remove rare gammas from Bremshtrahlung
    edep_cut = 1e-6
    mask_primary = (
        (gdf["parent"] == 0)
        & (gdf["edep"] > edep_cut)
        & (~gdf["trackprocess"].str.contains("eBrem", case=False, na=False))
        & (~gdf["process"].str.contains("transport", case=False, na=False))
        & (~gdf["process"].str.contains("UserSpecialCut", case=False, na=False))
    )
    gdf_primary = gdf[mask_primary]
    gdf_primary = gdf_primary.sort_values(["event", "time"]).groupby("event").first().reset_index()

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_raw_gamma_depth_validation(gdf_primary, plot_dir)

    df_doi_truth = df_doi.merge(
        gdf_primary[
            [
                "event",
                "x_gamma",
                "y_gamma",
                "z_gamma",
                "process",
                "edep",
                "delta_ke",
                "incident_energy",
                "incident_angle_deg",
            ]
        ],
        on="event",
        how="left",
    )
    df_doi_truth = df_doi_truth.rename(
        columns={
            "edep": "gamma_edep",
            "delta_ke": "gamma_delta_ke",
            "incident_energy": "gamma_incident_energy",
            "incident_angle_deg": "gamma_incident_angle_deg",
        }
    )
    df_doi_truth = df_doi_truth.dropna()

    plt.figure(figsize=(6, 6))
    plt.scatter(df_doi_truth["x_gamma"], df_doi_truth["y_gamma"], s=6, alpha=0.4)
    pixel_edges = np.arange(n + 1) * pitch - offset - pix / 2.0
    grid_min = pixel_edges[0]
    grid_max = pixel_edges[-1]
    for edge in pixel_edges:
        plt.plot([edge, edge], [grid_min, grid_max], color="black", linewidth=0.7, alpha=0.45)
        plt.plot([grid_min, grid_max], [edge, edge], color="black", linewidth=0.7, alpha=0.45)
    plt.plot([grid_min, grid_min], [grid_min, grid_max], color="black", linewidth=1.2, alpha=0.8)
    plt.plot([grid_max, grid_max], [grid_min, grid_max], color="black", linewidth=1.2, alpha=0.8)
    plt.plot([grid_min, grid_max], [grid_min, grid_min], color="black", linewidth=1.2, alpha=0.8)
    plt.plot([grid_min, grid_max], [grid_max, grid_max], color="black", linewidth=1.2, alpha=0.8)
    plt.xlabel("x_gamma [mm]")
    plt.ylabel("y_gamma [mm]")
    plt.title("Primary gamma interaction positions")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(plot_dir / "gamma-xy-scatter.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(df_doi_truth["z_gamma"], bins=40, alpha=0.85)
    plt.xlabel("z_gamma [mm]")
    plt.ylabel("Counts")
    plt.title("Primary gamma interaction z positions")
    plt.tight_layout()
    plt.savefig(plot_dir / "gamma-z-hist.png")
    plt.close()

    incident_energy = df_doi_truth["gamma_incident_energy"].to_numpy()
    incident_angle = df_doi_truth["gamma_incident_angle_deg"].to_numpy()
    finite_incident = np.isfinite(incident_energy) & np.isfinite(incident_angle)
    if np.any(finite_incident):
        incident_energy_plot = incident_energy[finite_incident]
        incident_angle_plot = incident_angle[finite_incident]
        energy_label = "gamma incident energy"
        # OpenGATE commonly stores kinetic energies in MeV. Convert to keV for
        # readability when the values look MeV-like.
        if np.nanmedian(incident_energy_plot) < 10.0:
            incident_energy_plot = 1000.0 * incident_energy_plot
            energy_label = "gamma incident energy [keV]"
        else:
            energy_label = "gamma incident energy"

        plt.figure(figsize=(6.5, 5))
        hb = plt.hexbin(
            incident_angle_plot,
            incident_energy_plot,
            gridsize=60,
            bins="log",
            mincnt=1,
            cmap="viridis",
        )
        plt.colorbar(hb, label="log10(counts)")
        plt.xlabel("incident angle to pixel normal [deg]")
        plt.ylabel(energy_label)
        plt.title("True incident gamma energy vs incident angle")
        plt.tight_layout()
        plt.savefig(plot_dir / "gamma-energy-vs-incident-angle.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(incident_energy_plot, bins=60, alpha=0.85)
        plt.xlabel(energy_label)
        plt.ylabel("Counts")
        plt.title("True incident gamma energy")
        plt.tight_layout()
        plt.savefig(plot_dir / "gamma-incident-energy-hist.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(incident_angle_plot, bins=60, alpha=0.85)
        plt.xlabel("incident angle to pixel normal [deg]")
        plt.ylabel("Counts")
        plt.title("True incident gamma angle")
        plt.tight_layout()
        plt.savefig(plot_dir / "gamma-incident-angle-hist.png")
        plt.close()

        plt.figure(figsize=(6.5, 5))
        plt.scatter(
            df_doi_truth.loc[finite_incident, "gamma_incident_angle_deg"],
            df_doi_truth.loc[finite_incident, "total_pe"],
            s=6,
            alpha=0.35,
        )
        plt.xlabel("incident angle to pixel normal [deg]")
        plt.ylabel("Total PE across all SiPM pixels")
        plt.title("Total PE vs incident gamma angle")
        plt.tight_layout()
        plt.savefig(plot_dir / "total-pe-vs-incident-angle.png")
        plt.close()

    return df_doi_truth


def main():
    args = parse_args()

    print("build npe df")
    df_npe_pixels = build_npe_df(args.sipm_root, rng_seed=args.rng_seed, step_size=args.step_size)

    print("build doi df")
    pe_window = None
    if args.total_pe_min is not None or args.total_pe_max is not None:
        pe_window = (args.total_pe_min, args.total_pe_max)
        print(f"Applying analysis-stage total_pe window: {pe_window}")
    else:
        print("No analysis-stage total_pe window; writing all events with SiPM signal")
    df_doi = build_doi_df(df_npe_pixels, pe_window=pe_window, plot_dir=args.plot_dir)

    print("build doi truth")
    df_doi_truth = build_truth_df(df_doi, args.gamma_root,  plot_dir=args.plot_dir )

    print("to csv")
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_doi_truth.to_csv(output_csv, index=False)

    print("done")


if __name__ == "__main__":
    main()
