#analyse.py
#this file:
# reads in root files with output of gamma scintillation and optical transport simulation
# assigns each optical photon to a virtual sipm pixel, applies a PDE, and estimates n photoelectrons in each sipm pixel
# applies energy window cut to max n_photoelectrons
# makes features dataframe: high-level features, per-pixel n photoelectrons 
# adds truth-level interaction coordinates from separate output file by matching with Event ID
# write resulting dataframe to disk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import uproot

# Geometry
pix = 3.0   # mm
foil = 0.2  # mm
n = 8
pitch = pix + foil
offset = (n - 1) * pitch / 2.0

# SiPM model (approximate: noise-free, saturation free)
PDE = 0.30
rng_seed = 123

#Read optical sim file and build dataframe
filename = "sipm_hits.root"
USE_STREAMING = True
STEP_SIZE = 500_000
ENERGY_WINDOW = (620.0, 1200.0)  # filter scatter events; set to None to disable
MAX_PE_FIT_WINDOW = (700.0, 1400.0)  # fit window for max_pe histogram; set to None to fit full range

def _find_tree_key(root_file):
    with uproot.open(root_file) as f:
        classnames = f.classnames()
        tree_key = next((k for k, v in classnames.items() if v.endswith('TTree')), None)
        if tree_key is None:
            raise RuntimeError(f'No TTree found in {root_file}. Keys: {list(classnames.keys())}')
        return tree_key

def _gauss_plus_falling_exp(x, A, mu, sigma, B, lamb, x0):
    gauss = A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    expo = B * np.exp(-lamb * (x - x0))
    return gauss + expo


def build_npe_df(filename, step_size=500_000):
    root_file = Path(filename)

    if not root_file.exists():
        roots = sorted(Path('.').glob('*.root'))
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
    df_counts['n_pe'] = rng.binomial(df_counts['n_photons'].to_numpy(), PDE)

    return df_counts

def build_doi_df(df_npe_pixels, pe_window):
    # Features: total PE, max-pixel fraction, spatial spread
    features = []

    for evt, grp in df_npe_pixels.groupby('event'):
        total_pe = grp['n_pe'].sum()
        if total_pe == 0:
            continue

        max_pe = grp['n_pe'].max()
        frac_max = max_pe / total_pe

        # light spread (RMS of pixel positions weighted by PE), is this the correct measure of spread?
        xs = grp['ix'].to_numpy()
        ys = grp['iy'].to_numpy()
        w = grp['n_pe'].to_numpy()
        cx = np.average(xs, weights=w)
        cy = np.average(ys, weights=w)
        spread = np.sqrt(np.average((xs - cx)**2 + (ys - cy)**2, weights=w))

        features.append({'event': evt, 'total_pe': total_pe, 'max_pe': max_pe, 'frac_max': frac_max, 'spread': spread})

    df_feat = pd.DataFrame(features)

    # Add per-pixel PE columns (wide format)
    pe_wide = df_npe_pixels.pivot_table(index='event', columns=['ix','iy'], values='n_pe', aggfunc='sum', fill_value=0)
    pe_wide.columns = [f'pe_{ix}_{iy}' for ix, iy in pe_wide.columns]
    pe_wide = pe_wide.reset_index()
    df_feat = df_feat.merge(pe_wide, on='event', how='left')

    #1-D histo of max pixel energy (to make sure energy window looks ok)
    plot_dir = Path("analyse-plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    counts, edges, _ = plt.hist(df_feat["max_pe"], bins=40, alpha=0.85)
    plt.xlabel("NPE in Max pixel")
    plt.ylabel("Counts")

    centers = 0.5 * (edges[:-1] + edges[1:])
    if len(centers) > 1 and np.any(counts > 0):
        fit_counts = counts
        fit_centers = centers
        fit_window = MAX_PE_FIT_WINDOW
        if fit_window is not None:
            lo, hi = fit_window
            fit_mask = np.ones_like(centers, dtype=bool)
            if lo is not None:
                fit_mask &= centers >= lo
            if hi is not None:
                fit_mask &= centers <= hi
            fit_counts = counts[fit_mask]
            fit_centers = centers[fit_mask]

        if len(fit_centers) > 0 and np.any(fit_counts > 0):
            x0 = float(fit_centers[0])
            A0 = float(fit_counts.max())
            mu0 = float(np.average(fit_centers, weights=fit_counts))
            sigma0 = float(df_feat["max_pe"].std()) if len(df_feat) > 1 else 1.0
            sigma0 = max(sigma0, 1.0)
            B0 = max(A0 * 0.2, 1.0)
            lamb0 = 1.0 / max(float(np.mean(fit_centers) - x0), 1.0)

            fit_ok = False
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
                fit_ok = True
            except Exception:
                fit_ok = False

            if fit_ok:
                fwhm = 2.355 * sigma_fit
                fwhm_pct = 100.0 * fwhm / mu_fit if mu_fit != 0 else np.nan
                xfit_lo = float(fit_centers[0])
                xfit_hi = float(fit_centers[-1])
                xfit = np.linspace(xfit_lo, xfit_hi, 400)
                gauss_fit = A_fit * np.exp(-0.5 * ((xfit - mu_fit) / sigma_fit) ** 2)
                exp_fit = B_fit * np.exp(-lamb_fit * (xfit - x0))
                total_fit = gauss_fit + exp_fit

                plt.plot(xfit, total_fit, "r-", linewidth=2, label="Gaussian + exponential")
                plt.plot(xfit, gauss_fit, "b--", linewidth=1.5, label="Gaussian component")
                plt.plot(xfit, exp_fit, "g--", linewidth=1.5, label="Exponential component")

                txt = (
                    f"$\\mu$ = {mu_fit:.3g}\n"
                    f"$\\sigma$ = {sigma_fit:.3g}\n"
                    f"FWHM = {fwhm:.3g}\n"
                    f"FWHM / $\\mu$ = {fwhm_pct:.2f}%"
                )
                plt.text(
                    0.02,
                    0.95,
                    txt,
                    transform=plt.gca().transAxes,
                    ha="left",
                    va="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    fontsize=9,
                )
                plt.legend(fontsize=8, loc="best")

    plt.tight_layout()
    plt.savefig(plot_dir / "max-pixel-energy-no-cuts.png")
    plt.close()

    if pe_window is not None:
        lo, hi = pe_window
        if lo is not None:
            df_feat = df_feat[df_feat['max_pe'] >= lo]
        if hi is not None:
            df_feat = df_feat[df_feat['max_pe'] <= hi]

    return df_feat

def build_truth_df(df_doi, filename):

    gamma_file = Path(filename)

    if not gamma_file.exists():
        print('gamma_steps.root not found; run module-sim.py with gamma_steps actor first.')
    else:
        with uproot.open(gamma_file) as f:
            classnames = f.classnames()
            tree_key = next((k for k, v in classnames.items() if v.endswith('TTree')), None)
            if tree_key is None:
                raise RuntimeError(f'No TTree found in {gamma_file}. Keys: {list(classnames.keys())}')
            tree = f[tree_key]
            g = tree.arrays(library='np')

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

        g_event = _gget('EventID')
        g_parent = _gget('ParentID')
        g_particle = _gget('ParticleName')
        g_process = _gget('ProcessDefinedStep')
        g_pos = _gget_vec('PostPosition') # note sure which postion defintion to take or if it matters
        if g_event is None or g_pos is None:
            raise RuntimeError('gamma_steps.root missing EventID or Position/PrePosition/PostPosition')

        # Build gamma steps dataframe
        gdf = pd.DataFrame({
            'event': g_event,
            'parent': g_parent if g_parent is not None else np.nan,
            'particle': g_particle.astype(str) if g_particle is not None else '',
            'process': g_process.astype(str) if g_process is not None else '',
            'x_gamma': g_pos[:, 0],
            'y_gamma': g_pos[:, 1],
            'z_gamma': g_pos[:, 2]
        })

        # Primary gamma first interaction: ParentID==0 and process != Transportation
        mask_primary = (gdf['parent'] == 0) & (~gdf['process'].str.contains('transport', case=False, na=False))
        gdf_primary = gdf[mask_primary]

        # First interaction per event
        gdf_primary = gdf_primary.groupby('event').first().reset_index()
        df_doi_truth = df_doi.merge(gdf_primary[['event','x_gamma','y_gamma','z_gamma', 'process']], on='event', how='left')
        df_doi_truth = df_doi_truth.dropna()

        plot_dir = Path("analyse-plots")
        plot_dir.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(6, 6))
        plt.scatter(df_doi_truth["x_gamma"], df_doi_truth["y_gamma"], s=6, alpha=0.4)
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

        return df_doi_truth

print('build npe df')

df_npe_pixels = build_npe_df(filename, step_size=STEP_SIZE)

print('build doi df ')

df_doi = build_doi_df(df_npe_pixels, pe_window=ENERGY_WINDOW)

print('build doi truth')

df_doi_truth = build_truth_df(df_doi, "gamma_steps.root")

print('to csv')

df_doi_truth.to_csv("df_training.csv", index=True)

print('done')
