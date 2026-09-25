#!/usr/bin/env python3
"""
Basic list-mode PET quicklook.

Input CSV columns (mm): x1,y1,z1,x2,y2,z2
Optional energy columns (keV): e1,e2 or E1,E2 or energy1,energy2 or edep1,edep2
Outputs PNG images using midpoint projection and x=0 intersection.
If energy columns are present, writes basic energy spectrum plots.
"""

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

from PIL import Image


Point = Tuple[float, float, float]
EventYZ = Tuple[float, float, float, float]


def read_events(path: str) -> Tuple[List[Point], List[Point], List[EventYZ], Optional[Dict[str, List[float]]]]:
    midpoints: List[Point] = []
    x0_points: List[Point] = []
    yz_events: List[EventYZ] = []
    energies: Dict[str, List[float]] = {"e1": [], "e2": [], "sum": []}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"x1", "y1", "z1", "x2", "y2", "z2"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"CSV must include columns: {', '.join(sorted(required))}. "
                f"Found: {reader.fieldnames}"
            )
        fieldnames = set(reader.fieldnames or [])
        energy_pair_candidates = [
            ("e1", "e2"),
            ("E1", "E2"),
            ("energy1", "energy2"),
            ("Energy1", "Energy2"),
            ("edep1", "edep2"),
            ("Edep1", "Edep2"),
            ("E1_keV", "E2_keV"),
            ("energy1_keV", "energy2_keV"),
            ("edep1_keV", "edep2_keV"),
        ]
        energy_single_candidates = [
            "energy_sum",
            "E_sum",
            "energy",
            "Energy",
            "E",
            "energy_keV",
            "E_keV",
        ]

        energy_pair = next(
            ((a, b) for a, b in energy_pair_candidates if a in fieldnames and b in fieldnames),
            None,
        )
        energy_single = next((c for c in energy_single_candidates if c in fieldnames), None)

        for row in reader:
            try:
                x1 = float(row["x1"])
                y1 = float(row["y1"])
                z1 = float(row["z1"])
                x2 = float(row["x2"])
                y2 = float(row["y2"])
                z2 = float(row["z2"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Bad row: {row}") from exc

            midpoints.append(((x1 + x2) * 0.5, (y1 + y2) * 0.5, (z1 + z2) * 0.5))
            yz_events.append((y1, z1, y2, z2))

            dx = x2 - x1
            if dx != 0.0:
                t = -x1 / dx
                if 0.0 <= t <= 1.0:
                    y = y1 + t * (y2 - y1)
                    z = z1 + t * (z2 - z1)
                    x0_points.append((0.0, y, z))

            if energy_pair is not None:
                try:
                    e1 = float(row[energy_pair[0]])
                    e2 = float(row[energy_pair[1]])
                except (TypeError, ValueError):
                    e1 = None
                    e2 = None
                if e1 is not None and e2 is not None:
                    energies["e1"].append(e1)
                    energies["e2"].append(e2)
                    energies["sum"].append(e1 + e2)
            elif energy_single is not None:
                try:
                    e_sum = float(row[energy_single])
                except (TypeError, ValueError):
                    e_sum = None
                if e_sum is not None:
                    energies["sum"].append(e_sum)

    if not energies["sum"]:
        energies = None
    return midpoints, x0_points, yz_events, energies


def compute_extent(points: List[Point], idx: int) -> Tuple[float, float]:
    vals = [p[idx] for p in points]
    return min(vals), max(vals)


def build_hist(
    points: List[Point],
    x_idx: int,
    y_idx: int,
    bin_mm: float,
    log_scale: bool = True,
    flip_y: bool = True,
) -> Tuple[Image.Image, Tuple[float, float, float, float], int]:
    if not points:
        raise ValueError("No points to histogram.")

    x_min, x_max = compute_extent(points, x_idx)
    y_min, y_max = compute_extent(points, y_idx)

    # Ensure non-zero span
    if x_max == x_min:
        x_max = x_min + bin_mm
    if y_max == y_min:
        y_max = y_min + bin_mm

    nx = int(math.floor((x_max - x_min) / bin_mm)) + 1
    ny = int(math.floor((y_max - y_min) / bin_mm)) + 1

    counts = [[0 for _ in range(nx)] for _ in range(ny)]
    for p in points:
        x = p[x_idx]
        y = p[y_idx]
        ix = int((x - x_min) / bin_mm)
        iy = int((y - y_min) / bin_mm)
        if 0 <= ix < nx and 0 <= iy < ny:
            row = (ny - 1 - iy) if flip_y else iy
            counts[row][ix] += 1

    max_count = max(max(row) for row in counts) if counts else 0
    if max_count == 0:
        max_count = 1

    flat = []
    if log_scale:
        denom = math.log1p(max_count)
        for row in counts:
            for c in row:
                flat.append(int(255 * math.log1p(c) / denom))
    else:
        for row in counts:
            for c in row:
                flat.append(int(255 * c / max_count))

    img = Image.new("L", (nx, ny))
    img.putdata(flat)
    extent = (x_min, x_max, y_min, y_max)
    return img, extent, max_count


def save_image(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)


def event_grid_extent(events: List[EventYZ]) -> Tuple[float, float, float, float]:
    y_vals: List[float] = []
    z_vals: List[float] = []
    for y1, z1, y2, z2 in events:
        y_vals.extend([y1, y2])
        z_vals.extend([z1, z2])
    return min(y_vals), max(y_vals), min(z_vals), max(z_vals)


def line_samples(
    y1: float,
    z1: float,
    y2: float,
    z2: float,
    step: float,
) -> List[Tuple[float, float, float]]:
    dy = y2 - y1
    dz = z2 - z1
    length = math.hypot(dy, dz)
    if length == 0.0:
        return [(y1, z1, 1.0)]
    n = max(1, int(math.ceil(length / step)))
    ds = length / n
    samples: List[Tuple[float, float, float]] = []
    for k in range(n + 1):
        t = k / n
        y = y1 + t * dy
        z = z1 + t * dz
        samples.append((y, z, ds))
    return samples


def mlem_yz(
    events: List[EventYZ],
    bin_mm: float,
    step_mm: float,
    iters: int,
    max_events: Optional[int] = None,
) -> Tuple[List[float], int, int, Tuple[float, float, float, float]]:
    if not events:
        raise ValueError("No events for MLEM.")

    if max_events is not None:
        events = events[:max_events]

    y_min, y_max, z_min, z_max = event_grid_extent(events)
    if y_max == y_min:
        y_max = y_min + bin_mm
    if z_max == z_min:
        z_max = z_min + bin_mm

    ny = int(math.floor((y_max - y_min) / bin_mm)) + 1
    nz = int(math.floor((z_max - z_min) / bin_mm)) + 1
    size = ny * nz

    def idx_from_yz(y: float, z: float) -> Optional[int]:
        iy = int((y - y_min) / bin_mm)
        iz = int((z - z_min) / bin_mm)
        if 0 <= iy < ny and 0 <= iz < nz:
            return (ny - 1 - iy) * nz + iz
        return None

    sensitivity = [0.0] * size
    for y1, z1, y2, z2 in events:
        for y, z, w in line_samples(y1, z1, y2, z2, step_mm):
            idx = idx_from_yz(y, z)
            if idx is not None:
                sensitivity[idx] += w

    image = [1.0 if s > 0.0 else 0.0 for s in sensitivity]
    eps = 1e-9

    for _ in range(iters):
        numerator = [0.0] * size
        for y1, z1, y2, z2 in events:
            ray = line_samples(y1, z1, y2, z2, step_mm)
            denom = 0.0
            for y, z, w in ray:
                idx = idx_from_yz(y, z)
                if idx is not None:
                    denom += w * image[idx]
            if denom <= 0.0:
                continue
            inv = 1.0 / denom
            for y, z, w in ray:
                idx = idx_from_yz(y, z)
                if idx is not None:
                    numerator[idx] += w * inv
        for i in range(size):
            if sensitivity[i] > 0.0:
                image[i] = image[i] * numerator[i] / (sensitivity[i] + eps)

    extent = (y_min, y_max, z_min, z_max)
    return image, nz, ny, extent


def image_from_float_grid(values: List[float], width: int, height: int) -> Image.Image:
    max_val = max(values) if values else 1.0
    if max_val <= 0.0:
        max_val = 1.0
    data = [int(255 * v / max_val) for v in values]
    img = Image.new("L", (width, height))
    img.putdata(data)
    return img


def write_summary(
    path: str,
    total_events: int,
    x0_events: int,
    bin_mm: float,
    mlem_iters: int,
    mlem_step_mm: float,
    mlem_max_events: Optional[int],
    details: List[Tuple[str, Tuple[float, float, float, float], float]],
    energy_stats: Optional[Dict[str, float]] = None,
) -> None:
    lines = [
        f"total_events: {total_events}",
        f"x0_events: {x0_events}",
        f"bin_mm: {bin_mm}",
        f"mlem_iters: {mlem_iters}",
        f"mlem_step_mm: {mlem_step_mm}",
        f"mlem_max_events: {mlem_max_events if mlem_max_events is not None else 'all'}",
        "",
    ]
    for name, extent, max_count in details:
        x_min, x_max, y_min, y_max = extent
        if isinstance(max_count, float) and not max_count.is_integer():
            max_str = f"{max_count:.6f}"
        else:
            max_str = f"{int(max_count)}"
        lines.extend(
            [
                f"{name}:",
                f"  x_min_mm: {x_min:.3f}",
                f"  x_max_mm: {x_max:.3f}",
                f"  y_min_mm: {y_min:.3f}",
                f"  y_max_mm: {y_max:.3f}",
                f"  max_count: {max_str}",
                "",
            ]
        )
    if energy_stats:
        lines.extend(
            [
                "energy:",
                f"  count: {int(energy_stats['count'])}",
                f"  min_keV: {energy_stats['min']:.3f}",
                f"  max_keV: {energy_stats['max']:.3f}",
                f"  mean_keV: {energy_stats['mean']:.3f}",
                "",
            ]
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Basic list-mode PET quicklook from two-module data."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to CSV with columns x1,y1,z1,x2,y2,z2 (mm).",
    )
    parser.add_argument(
        "--outdir",
        default="outputs/recon",
        help="Directory to write PNG outputs.",
    )
    parser.add_argument(
        "--bin-mm",
        type=float,
        default=1.0,
        help="Histogram bin size in mm (default: 1.0).",
    )
    parser.add_argument(
        "--mlem-iters",
        type=int,
        default=0,
        help="Run basic MLEM in the y-z plane for N iterations (default: 0).",
    )
    parser.add_argument(
        "--mlem-step-mm",
        type=float,
        default=0.5,
        help="Sampling step along LOR in mm (default: 0.5).",
    )
    parser.add_argument(
        "--mlem-max-events",
        type=int,
        default=None,
        help="Optional cap on number of events for MLEM.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable log scaling.",
    )
    parser.add_argument(
        "--energy-bins",
        type=int,
        default=200,
        help="Bins for energy spectra (default: 200).",
    )
    parser.add_argument(
        "--energy-min",
        type=float,
        default=None,
        help="Minimum energy for spectra (keV).",
    )
    parser.add_argument(
        "--energy-max",
        type=float,
        default=None,
        help="Maximum energy for spectra (keV).",
    )
    args = parser.parse_args()

    midpoints, x0_points, yz_events, energies = read_events(args.input)
    log_scale = not args.no_log

    outputs: List[Tuple[str, Tuple[float, float, float, float], float]] = []

    img, extent, max_count = build_hist(
        midpoints, 0, 1, args.bin_mm, log_scale=log_scale, flip_y=True
    )
    out_path = os.path.join(args.outdir, "mid_xy.png")
    save_image(img, out_path)
    outputs.append(("mid_xy.png", extent, max_count))

    img, extent, max_count = build_hist(
        midpoints, 0, 2, args.bin_mm, log_scale=log_scale, flip_y=True
    )
    out_path = os.path.join(args.outdir, "mid_xz.png")
    save_image(img, out_path)
    outputs.append(("mid_xz.png", extent, max_count))

    img, extent, max_count = build_hist(
        midpoints, 1, 2, args.bin_mm, log_scale=log_scale, flip_y=True
    )
    out_path = os.path.join(args.outdir, "mid_yz.png")
    save_image(img, out_path)
    outputs.append(("mid_yz.png", extent, max_count))

    if x0_points:
        img, extent, max_count = build_hist(
            x0_points, 1, 2, args.bin_mm, log_scale=log_scale, flip_y=True
        )
        out_path = os.path.join(args.outdir, "x0_yz.png")
        save_image(img, out_path)
        outputs.append(("x0_yz.png", extent, max_count))

    if args.mlem_iters > 0:
        image, width, height, extent = mlem_yz(
            yz_events,
            bin_mm=args.bin_mm,
            step_mm=args.mlem_step_mm,
            iters=args.mlem_iters,
            max_events=args.mlem_max_events,
        )
        mlem_img = image_from_float_grid(image, width, height)
        out_path = os.path.join(args.outdir, "mlem_yz.png")
        save_image(mlem_img, out_path)
        max_val = max(image) if image else 0.0
        outputs.append(("mlem_yz.png", extent, max_val))

    energy_stats = None
    if energies is not None:
        try:
            import matplotlib.pyplot as plt
        except Exception as exc:
            raise RuntimeError(
                "Energy columns detected, but matplotlib is not available. "
                "Install matplotlib or remove energy columns."
            ) from exc

        os.makedirs(args.outdir, exist_ok=True)

        def _plot_energy_hist(data: List[float], title: str, filename: str) -> None:
            if not data:
                return
            plt.figure(figsize=(6, 4))
            if args.energy_min is not None and args.energy_max is not None:
                hist_range = (args.energy_min, args.energy_max)
            else:
                hist_range = None
            plt.hist(data, bins=args.energy_bins, range=hist_range, color="#3b82f6")
            plt.xlabel("Energy (keV)")
            plt.ylabel("Counts")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(args.outdir, filename), dpi=150)
            plt.close()

        _plot_energy_hist(energies.get("sum", []), "Energy Sum", "energy_sum.png")
        _plot_energy_hist(energies.get("e1", []), "Energy 1", "energy1.png")
        _plot_energy_hist(energies.get("e2", []), "Energy 2", "energy2.png")

        if energies.get("sum"):
            vals = energies["sum"]
            energy_stats = {
                "count": float(len(vals)),
                "min": min(vals),
                "max": max(vals),
                "mean": sum(vals) / len(vals),
            }

    summary_path = os.path.join(args.outdir, "summary.txt")
    write_summary(
        summary_path,
        len(midpoints),
        len(x0_points),
        args.bin_mm,
        args.mlem_iters,
        args.mlem_step_mm,
        args.mlem_max_events,
        outputs,
        energy_stats,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
