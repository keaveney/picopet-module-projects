import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
try:
    import awkward as ak
except Exception:  # awkward is optional
    ak = None

root_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("hits.root")
if not root_file.exists():
    roots = sorted(Path(".").glob("*.root"))
    hint = f"Available .root files in cwd: {[p.name for p in roots]}" if roots else "No .root files found in cwd."
    raise FileNotFoundError(f"{root_file} not found. {hint}")

with uproot.open(root_file) as f:
    classnames = f.classnames()
    tree_key = next((k for k, v in classnames.items() if v.endswith("TTree")), None)
    if tree_key is None:
        raise RuntimeError(f"No TTree found in {root_file}. Keys: {list(classnames.keys())}")
    tree = f[tree_key]
    arrays = tree.arrays(library="np")

def _get(name):
    return arrays[name] if name in arrays else None

def _missing(n, value=np.nan):
    return np.full(n, value)

n = len(arrays[next(iter(arrays.keys()))]) if len(arrays) > 0 else 0

pos = _get("PostPosition")
if pos is None:
    pos = _get("PrePosition")
if pos is None:
    x_mm = y_mm = z_mm = _missing(n)
else:
    x_mm, y_mm, z_mm = pos[:, 0], pos[:, 1], pos[:, 2]

edep = _get("TotalEnergyDeposit")
if edep is None:
    edep = _get("Edep")
edep_keV = edep * 1000.0 if edep is not None else _missing(n)  # assuming MeV -> keV

process = _get("ProcessName")
if process is None:
    process = _get("TrackCreatorProcess")

volume = _get("TrackVolumeName")
if volume is None:
    volume = _get("VolumeName")

df = pd.DataFrame(
    {
        "event": _get("EventID") if _get("EventID") is not None else _missing(n),
        "run": _get("RunID") if _get("RunID") is not None else _missing(n),
        "thread": _get("ThreadID") if _get("ThreadID") is not None else _missing(n),
        "track": _get("TrackID") if _get("TrackID") is not None else _missing(n),
        "parent": _get("ParentID") if _get("ParentID") is not None else _missing(n),
        "particle": _get("ParticleName").astype(str) if _get("ParticleName") is not None else _missing(n, ""),
        "process": process.astype(str) if process is not None else _missing(n, ""),
        "edep_keV": edep_keV,
        "x_mm": x_mm,
        "y_mm": y_mm,
        "z_mm": z_mm,
        "t_ns": _get("GlobalTime") if _get("GlobalTime") is not None else _missing(n),
        "volume": volume.astype(str) if volume is not None else _missing(n, ""),
        "copyno": _get("CopyNo") if _get("CopyNo") is not None else _missing(n),
    }
)

# Keep only pixel hits by default (drop ESR / other volumes)
pixel_mask = df["volume"].astype(str).str.startswith("pixel_")
if pixel_mask.any():
    dropped = (~pixel_mask).sum()
    if dropped > 0:
        print(f"Filtered out {dropped} non-pixel hits")
    df = df[pixel_mask].reset_index(drop=True)
else:
    print("WARNING: No pixel hits found; keeping all hits.")

# Classify interaction type
df["interaction_type"] = np.where(
    df["process"].str.contains("phot", case=False, na=False),
    "PE",
    np.where(df["process"].str.contains("compt", case=False, na=False), "Compton", "Other"),
)

# -------------------------
# Plot 1: Histogram of deposited energy per hit
# -------------------------
edep_vals = df["edep_keV"].to_numpy()
edep_vals = edep_vals[np.isfinite(edep_vals)]
edep_vals = edep_vals[edep_vals > 0]

plt.figure(figsize=(6, 4))
plt.hist(edep_vals, bins=100, color="#1f77b4", alpha=0.85)
plt.xlabel("Deposited energy per hit (keV)")
plt.ylabel("Counts")
plt.title("Hit Energy Deposition")
plt.tight_layout()
plt.savefig("edep_hist.png", dpi=200)
plt.close()

# -------------------------
# Plot 2: 2D heatmap of total deposited energy per pixel
# -------------------------
pixel_re = re.compile(r"pixel_(\d+)_(\d+)")

def _extract_pixel(name):
    m = pixel_re.search(name)
    if not m:
        return np.nan, np.nan
    return float(m.group(1)), float(m.group(2))

vol_names = df["volume"].astype(str)
ix_iy = vol_names.apply(_extract_pixel)
df["ix"] = [v[0] for v in ix_iy]
df["iy"] = [v[1] for v in ix_iy]

df_pix = df[np.isfinite(df["ix"]) & np.isfinite(df["iy"])].copy()
if len(df_pix) == 0:
    print("No pixel volume names found; skipping pixel heatmap.")
else:
    max_ix = int(df_pix["ix"].max())
    max_iy = int(df_pix["iy"].max())
    grid = np.zeros((max_iy + 1, max_ix + 1), dtype=float)

    for _, row in df_pix.iterrows():
        if np.isfinite(row["edep_keV"]) and row["edep_keV"] > 0:
            grid[int(row["iy"]), int(row["ix"])] += row["edep_keV"]

    plt.figure(figsize=(5, 5))
    im = plt.imshow(grid, origin="lower", cmap="magma")
    plt.colorbar(im, label="Total deposited energy (keV)")
    plt.xlabel("pixel ix")
    plt.ylabel("pixel iy")
    plt.title("Total Edep per Pixel")
    plt.tight_layout()
    plt.savefig("pixel_edep_map.png", dpi=200)
    plt.close()

print("Wrote edep_hist.png and pixel_edep_map.png")

# -------------------------
# Event-level tree with classification
# -------------------------
def _primary_mask(grp):
    if "particle" in grp.columns and grp["particle"].notna().any():
        # prefer explicit gamma tag when available
        return grp["particle"].astype(str).str.lower().eq("gamma")
    if "parent" in grp.columns and grp["parent"].notna().any():
        return grp["parent"] == 0
    if "track" in grp.columns and grp["track"].notna().any():
        return grp["track"] == 1
    return pd.Series([False] * len(grp), index=grp.index)


def _first_interaction(grp):
    g = grp.copy()
    if "t_ns" in g.columns:
        g = g.sort_values("t_ns", kind="mergesort")
    return g.iloc[0] if len(g) else None


def _classify_event_first_proc(grp):
    proc = grp["process"].astype(str).str.lower().fillna("")
    prim = _primary_mask(grp)
    use = grp[prim] if prim.any() else grp
    if len(use) == 0:
        return "", "Other"
    first = _first_interaction(use)
    if first is None:
        return "", "Other"
    first_proc = str(first.get("process", "")).lower()
    if "phot" in first_proc:
        return first_proc, "PE"
    if "compt" in first_proc:
        return first_proc, "CS"
    if "rayl" in first_proc:
        return first_proc, "Rayleigh"
    if "transport" in first_proc:
        return first_proc, "Transportation"
    return first_proc, "Other"


def _classify_event_cspe(grp):
    proc = grp["process"].str.lower().fillna("")
    t_pe = grp.loc[proc.str.contains("phot"), "t_ns"].min()
    t_cs = grp.loc[proc.str.contains("compt"), "t_ns"].min()
    has_pe = np.isfinite(t_pe)
    has_cs = np.isfinite(t_cs)
    if has_pe and (not has_cs or t_pe <= t_cs):
        return "PE"
    if has_cs:
        if has_pe and t_pe > t_cs:
            return "CS-PE"
        return "CS-NOPE"
    return "Other"

df_ev = df[np.isfinite(df["event"])].copy()
if len(df_ev) == 0:
    print("No EventID found; skipping Events tree.")
elif ak is None:
    print("awkward is not installed; skipping Events tree.")
else:
    df_ev["event"] = df_ev["event"].astype(np.int64)
    df_ev = df_ev.sort_values(["event", "t_ns"], kind="mergesort")

    event_ids = []
    classes = []
    total_edep = []
    hit_edep = []
    hit_t = []
    hit_x = []
    hit_y = []
    hit_z = []
    hit_ix = []
    hit_iy = []
    # Note: uproot cannot write jagged string arrays; keep only numeric branches here.

    for event_id, grp in df_ev.groupby("event", sort=True):
        event_ids.append(event_id)
        first_proc, cls_first = _classify_event_first_proc(grp)
        classes.append(cls_first)
        total_edep.append(np.nansum(grp["edep_keV"].to_numpy()))
        hit_edep.append(grp["edep_keV"].to_list())
        hit_t.append(grp["t_ns"].to_list())
        hit_x.append(grp["x_mm"].to_list())
        hit_y.append(grp["y_mm"].to_list())
        hit_z.append(grp["z_mm"].to_list())
        hit_ix.append(grp["ix"].to_list())
        hit_iy.append(grp["iy"].to_list())

    events_tree = {
        "EventID": np.array(event_ids, dtype=np.int64),
        "Class": ak.Array(classes),
        "TotalEdep_keV": np.array(total_edep, dtype=np.float64),
        "HitEdep_keV": ak.Array(hit_edep),
        "HitTime_ns": ak.Array(hit_t),
        "HitX_mm": ak.Array(hit_x),
        "HitY_mm": ak.Array(hit_y),
        "HitZ_mm": ak.Array(hit_z),
        "HitIx": ak.Array(hit_ix),
        "HitIy": ak.Array(hit_iy),
    }

    try:
        with uproot.update(str(root_file)) as f:
            f["Events"] = events_tree
        print(f"Wrote Events tree into {root_file}")
    except Exception as exc:
        out = root_file.with_name(f\"{root_file.stem}_events.root\")
        with uproot.recreate(str(out)) as f:
            f[\"Events\"] = events_tree
        print(f\"Could not update {root_file} ({exc}); wrote {out} instead\")
