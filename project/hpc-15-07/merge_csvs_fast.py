#!/usr/bin/env python3
"""Fast, constant-memory merge of per-job training CSV files.

This deliberately avoids pandas.  If all input CSV files have the same header,
merging is just a text concatenation problem: write the header once, then append
the data rows from each file while skipping its header line.
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge many per-job df_training.csv files without loading them into memory."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/scratch/USER/module-sim"),
        help="Directory containing job subdirectories. Default: /scratch/USER/module-sim.",
    )
    parser.add_argument(
        "--pattern",
        default="job_*/df_training.csv",
        help='Glob pattern below --input-dir. Default: "job_*/df_training.csv".',
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/scratch/USER/module-sim/df_training_merged.csv"),
        help="Merged output CSV path. Default: /scratch/USER/module-sim/df_training_merged.csv.",
    )
    parser.add_argument(
        "--buffer-mb",
        type=int,
        default=64,
        help="Copy buffer size in MB. Larger can help on fast filesystems. Default: 64.",
    )
    parser.add_argument(
        "--allow-header-mismatch",
        action="store_true",
        help="Merge even if later CSV headers differ from the first header.",
    )
    parser.add_argument(
        "--count-rows",
        action="store_true",
        help="Count merged data rows while copying. This adds some overhead.",
    )
    return parser.parse_args()


def find_inputs(input_dir: Path, pattern: str, output: Path):
    input_dir = input_dir.expanduser().resolve()
    output = output.expanduser().resolve()
    files = []
    for path in sorted(input_dir.glob(pattern)):
        path = path.resolve()
        if path == output:
            continue
        if path.is_file() and path.stat().st_size > 0:
            files.append(path)
    return files


def copy_data_rows(fin, fout, buffer_size: int, count_rows: bool):
    """Copy all remaining bytes from fin to fout after the header has been read."""
    rows = 0
    bytes_copied = 0
    last_byte = None
    while True:
        chunk = fin.read(buffer_size)
        if not chunk:
            break
        bytes_copied += len(chunk)
        last_byte = chunk[-1:]
        if count_rows:
            rows += chunk.count(b"\n")
        fout.write(chunk)

    if bytes_copied > 0 and last_byte != b"\n":
        # Avoid concatenating the last row of one file with the first row of the
        # next file if a job-produced CSV is missing its final newline.
        fout.write(b"\n")
        if count_rows:
            rows += 1

    return rows if count_rows else None


def merge_csvs(files, output: Path, buffer_size: int, allow_header_mismatch: bool, count_rows: bool):
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(output.name + ".tmp")

    first_header = None
    files_written = 0
    rows_written = 0 if count_rows else None
    skipped_header_mismatch = 0

    with tmp_output.open("wb") as fout:
        for csv_path in files:
            with csv_path.open("rb") as fin:
                header = fin.readline()
                if not header:
                    continue

                if first_header is None:
                    first_header = header
                    fout.write(header)
                elif header != first_header:
                    message = (
                        f"Header mismatch in {csv_path}\n"
                        f"First header: {first_header.decode(errors='replace').strip()}\n"
                        f"This header:  {header.decode(errors='replace').strip()}"
                    )
                    if not allow_header_mismatch:
                        raise RuntimeError(message)
                    print(f"WARNING: {message}", file=sys.stderr)
                    skipped_header_mismatch += 1

                copied_rows = copy_data_rows(fin, fout, buffer_size, count_rows)
                if count_rows:
                    rows_written += int(copied_rows)
                files_written += 1

    if first_header is None:
        tmp_output.unlink(missing_ok=True)
        raise RuntimeError("No non-empty input CSV files found")

    tmp_output.replace(output)
    return {
        "files_written": files_written,
        "rows_written": rows_written,
        "header_mismatches_allowed": skipped_header_mismatch,
        "output": output,
        "output_size_bytes": output.stat().st_size,
    }


def main():
    args = parse_args()
    start = time.perf_counter()
    buffer_size = max(1, args.buffer_mb) * 1024 * 1024

    files = find_inputs(args.input_dir, args.pattern, args.output)
    print(f"Found {len(files)} non-empty CSV files")
    if len(files) == 0:
        raise SystemExit("No input files to merge")

    result = merge_csvs(
        files,
        args.output,
        buffer_size=buffer_size,
        allow_header_mismatch=args.allow_header_mismatch,
        count_rows=args.count_rows,
    )
    elapsed = time.perf_counter() - start

    print(f"Merged {result['files_written']} files -> {result['output']}")
    if result["rows_written"] is not None:
        print(f"Merged data rows: {result['rows_written']}")
    print(f"Output size: {result['output_size_bytes'] / 1e9:.3f} GB")
    print(f"Elapsed: {elapsed:.1f} s")


if __name__ == "__main__":
    main()
