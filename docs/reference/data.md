# Data and script map

## Which input belongs to which exercise?

All paths below are relative to the extracted package root.

| Input | Contents | Used for | Provenance limit |
| --- | --- | --- | --- |
| `project/hpc/test.csv` | 2,082 feature/truth rows, 64 PE channels | Practicals 1–2 | Historical producing configuration unknown |
| `project/hpc/local-cnn-zgmm-plain-test/test-predictions-with-uncertainty.csv` | First 1,024 rows of a historical 300,000-row K = 8 prediction file | Practicals 3–5 | Not guaranteed random; lacks incident energy/angle metadata |
| `project/hpc-15-07/cnn-gmm-n{1,4,8,16}-50M/` | Saved metrics and selected figures | Practical 6 | Full configurations, prediction CSVs and weights absent |
| `project/hpc/local-vis-test/*.root` | Tiny historical ROOT pair | Optional later reference only | Separate example, not the source of the other samples |

The older `hpc/` directory supplies inputs; **all current scientific commands use `hpc-15-07/`**. These examples must not be joined into an invented end-to-end provenance chain.

## Core feature schema

| Column | Meaning | Unit / caveat |
| --- | --- | --- |
| `event` | Within-job EventID | Can repeat across jobs |
| `pe_ix_iy` | Detected photoelectron count in virtual channel | 64 expected channels in the supplied feature sample |
| `total_pe`, `max_pe` | Sum and largest channel count | Counts |
| `pe_3x3_max` | Clipped 3 × 3 sum around the maximum channel | Counts; availability depends on file/derived inputs |
| `frac_max` | `max_pe / total_pe` | Dimensionless |
| `spread` | PE-weighted RMS transverse distance from centroid | Pixel-index units |
| `x_gamma`, `y_gamma`, `z_gamma` | First retained energy-depositing primary-gamma step position | mm; z centred in crystal |
| `gamma_incident_energy` | Pre-step gamma energy | Normally MeV |
| `gamma_delta_ke` | Pre-step minus post-step gamma energy | Normally MeV |
| `gamma_edep` | Local selected-step deposit | Normally MeV, not total event deposit |
| `gamma_incident_angle_deg` | acos of absolute step-direction z component | Degrees; folded onto 0–90° |
| `process` | Selected step process, e.g. `phot`, `compt` | `phot` means photoelectric, not photoelectron |

## Saved prediction schema

`x_true`, `y_true`, `z_true`; `x_pred`, `y_pred`, `z_pred`; and `x_sigma`, `y_sigma`, `z_sigma` use mm. Depth-mixture groups are `z_gmm_weight_k` (dimensionless), `z_gmm_mu_k` (mm), `z_gmm_sigma_k` (mm), with zero-based k. `z_gmm_mean` stores the mixture mean when present.

The evaluator detects prediction columns. If they are absent it takes the validation-only route. Multi-estimator evaluation writes `z-gmm-mean/`, `z-gmm-median/`, `z-gmm-mode/`; single-estimator evaluation writes directly into its output directory.

## Authoritative entry points

Paths in this table are under `project/hpc-15-07/`.

| Script | Purpose | Initial local route? |
| --- | --- | --- |
| `evaluate_cnn_performance.py` | CSV validation and saved-prediction diagnostics | Yes |
| `html_report.py` | Sibling helper producing the local gallery | Yes; keep beside evaluator |
| `plot_gmm_component_performance.py` | Render saved per-estimator metric JSONs | Yes |
| `module-sim-doi-511-cone-array.py` | GATE source, geometry and scoring | Later |
| `analyse-array.py` | ROOT to counts, light observables and truth CSV | Later; normally in simulation job |
| `merge_csvs_fast.py` | Concatenate compatible job CSVs | Later |
| `train-cnn-array-xyz-gmm.py` | Train density model and export test predictions | Later |

## Useful evaluator switches

| Switch | What it controls |
| --- | --- |
| `--predictions-csv` | Input CSV, including feature-only validation inputs |
| `--outdir` | Output directory; choose a new one for distinct trials |
| `--z-gmm-point-estimators mean median mode` | Estimate(s) recomputed from saved density parameters |
| `--no-per-pixel-plots` | Skip per-pixel plots and channel loading |
| `--n-z-resolution-bins` | Number of true-z slices |
| `--n-pred-z-condition-bins` | Predicted-z conditioning bins |
| `--n-pred-z-condition-gmm-components` | Components in a separate histogram fit, not the CNN K |

For exact options, run `python evaluate_cnn_performance.py --help` in its directory. JSON metrics may contain Python-style `NaN`/`Infinity`; strict JSON clients may reject them. Python’s standard reader accepts these files.
