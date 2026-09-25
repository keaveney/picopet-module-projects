# 0 · Set up for local work

**Goal:** prepare a Python environment for the supplied CSV exercises. Keep the extracted package together; commands below assume its root is the directory containing `mkdocs.yml`, `project/` and `README.md`.

## Why am I doing this?

The same scripts used in the research workflow will produce your plots. Consistent dependencies make numerical and plotting differences easier to diagnose. You need no trained checkpoint, GPU, ROOT processing or cluster login for the initial route.

!!! info "Setup status: clean student installation untested"
    The supplied `project/requirements-local.txt` records the researcher’s working **Python 3.12.4** environment. It is an installed-package snapshot, not a universal cross-platform lockfile. Check a fresh installation on your machine before starting the exercises; see the [reproducibility notes](../reference/data-checks.md#environment-and-reproducibility).

## What do I run?

Install/select Python 3.12.4 to match the recorded environment. Check the patch version: `python3.12` can resolve to another Python 3.12 release.

**macOS/Linux terminal — working directory: extracted package root**

```bash
python3.12 --version
python3.12 -m venv .venv-analysis
source .venv-analysis/bin/activate
python -m pip install -r project/requirements-local.txt
python -m pip check
python -c "import numpy, pandas, matplotlib, scipy; print('CSV analysis imports OK')"
```

**Windows PowerShell — equivalent setup, untested**

```powershell
py -3.12 --version
py -3.12 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install -r project/requirements-local.txt
.\.venv-analysis\Scripts\python.exe -m pip check
.\.venv-analysis\Scripts\python.exe -c "import numpy, pandas, matplotlib, scipy; print('CSV analysis imports OK')"
```

For subsequent exercises in PowerShell, invoke the virtual environment’s Python by its path and enter each command on one line; the displayed `\` continuation style is for macOS/Linux shells. From `project/hpc-15-07`, that interpreter path is `..\..\.venv-analysis\Scripts\python.exe`.

The snapshot includes PyTorch and uproot because it covers the broader local environment. **Evaluating saved CSVs imports neither PyTorch nor GATE.** Do not replace the snapshot with an old simulation environment or silently change its pins if installation fails; record the platform and exact error.

## What should I see?

The version command reports the interpreter in use. `pip check` should report no broken requirements, followed by `CSV analysis imports OK`. If installation fails, it has not passed setup, even if some imports happen to work in another environment.

## How do I know it worked?

Record the Python version, operating system and `pip check` output. Then perform [the input consistency checks](features.md). They test the data and active interpreter together.

## Where do results go?

Run the scientific commands from **`project/hpc-15-07`**:

```bash
cd project/hpc-15-07
```

Subsequent exercises write into `student-runs/` under that folder, leaving supplied scripts and historical results untouched. Use a fresh output folder when changing a selection so old plots are not mistaken for current results. The evaluator already uses Matplotlib’s noninteractive `Agg` backend.

## Common failures

| Symptom | Action |
| --- | --- |
| `python3.12` not found | Select/install the required interpreter; do not assume your default Python is suitable |
| A pinned package has no compatible distribution | Save the complete error and platform details; ask the supervisor to approve a tested environment adjustment |
| Imports work in VS Code but fail in the terminal | Check the selected VS Code interpreter and the terminal virtual environment |
| NumPy/PyTorch ABI error | Check that the active environment matches the supplied NumPy 1.26.4 / PyTorch 2.2.2 versions before changing dependencies |
| Matplotlib cache permission error | Select a writable cache: `export MPLCONFIGDIR="$PWD/.mpl-cache"` in a POSIX shell |

**Check your understanding:** why does a recorded working environment provide stronger evidence than a list of plausible package names, but weaker evidence than a clean installation on your own computer?

<div class="next" markdown>Next: [read a light map →](features.md)</div>
