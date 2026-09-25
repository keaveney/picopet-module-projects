# Troubleshooting

Check the command, working directory, input and active interpreter first. Keep the original source intact while diagnosing a problem.

| Problem | Likely explanation / next step |
| --- | --- |
| `No module named html_report` | Run the authoritative evaluator with its sibling `html_report.py`; do not move the script alone |
| File not found | Commands assume `project/hpc-15-07`; inputs use `../hpc/` relative to it |
| Prediction columns not found | Expected for the feature CSV: the script runs validation-only plots |
| No angle plot | The saved prediction excerpt has no incident-angle metadata |
| No narrow-energy selection | The excerpt lacks incident energy, or fewer than 50 feature rows pass the requested window |
| A fitted point is absent | Check `fit_ok` and populated bins; the comparison filters unsuccessful fitted points |
| `pull-z.png` has mm on its axis | It contains a dimensional residual despite the historical filename |
| Some neighbourhood plots are missing | `--no-per-pixel-plots` also suppresses PE channel loading |
| Mean/median/mode folders missing | A single estimator writes directly into `--outdir`; multiple estimators create subfolders |
| Comparison finds no metrics | Use the supplied `cnn-gmm-n*-50M/z-gmm-*` directory structure |
| Different numerical fits on another machine | Compare versions, inputs and options; these small-sample fits can be sensitive to software and statistics |
| A generic JSON viewer rejects metrics | Files can contain NaN/Infinity; inspect using Python’s `json` module |
| Browser search does not work after double-clicking HTML | Serve `site/` over localhost; file URLs can block the search worker |

## Before asking for help

Send the full command, working directory, Python version, input filename/hash, complete error message and whether any outputs were written. Include the relevant `fit_ok` and selection fields if the issue is scientific interpretation rather than program execution.

Do not resolve installation trouble by silently changing scientific code or dependencies. Do not use an old cluster driver as a workaround for a local CSV exercise.
