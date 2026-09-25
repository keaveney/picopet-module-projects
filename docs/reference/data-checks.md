# Example data checks

Use these checks alongside the [data and script map](data.md). They describe the supplied teaching inputs, rather than a production performance benchmark. Recheck inputs and fit flags when changing data, selections or software versions.

## Feature CSV

`project/hpc/test.csv` contains **2,082 rows, 78 columns and all 64 expected channel columns**. Channel counts are finite, nonmissing and nonnegative. The largest difference between the channel sum and `total_pe` is zero. The z truth coordinates range from approximately −7.495 to +7.496 mm.

[Practical 1](../practicals/features.md) shows how to inspect the schema, counts and truth bounds.

## Saved prediction excerpt

The supplied prediction excerpt contains **1,024 rows and K = 8**. Mixture weights are finite and nonnegative; their largest sum-to-one deviation is approximately **2.25 × 10⁻⁷**. Component widths are finite and positive. See [Practical 3](../practicals/predictions.md) for the input checks.

Changing only the z point estimator should leave all-event x/y MAE and RMSE unchanged. Check this when [comparing mean, median and mode](../practicals/estimators.md).

## Fit and metadata limitations

- The feature example contains **814 rows** in the requested selected-step 511 ± 1 keV energy window.
- The example total-PE, visible-energy and attenuation fits report success. The maximum-channel spectrum reports **`fit_ok = false`**; its fallback must not be quoted as a successful fitted measurement.
- The prediction excerpt lacks incident-angle and incident-energy metadata. The angle plot is skipped, and a narrow-energy attenuation selection cannot be established for that input.
- Sparse-bin fit flags must be inspected in individual diagnostics. A successful program exit does not imply that every fit succeeded.

[Practical 2](../practicals/validation.md) shows how to inspect the selection fields and fit flags in your own output. Numerical fits can vary with software and sample statistics.

## Saved component-count comparisons

The comparison covers four component counts, three estimators and three coordinates: **36 table rows**, with six output figures. [Practical 6](../practicals/components.md) describes the checks and the limits of comparing historical models without their full configurations and matched data splits.

## Environment and reproducibility

The supplied `project/requirements-local.txt` records the researcher’s Python 3.12.4 environment. A clean installation of that snapshot on each student’s machine still needs checking. The initial CSV exercises do not import PyTorch or GATE; their success alone cannot establish that training or simulation dependencies work.

Record the interpreter and package versions, input identities, exact commands and output selections when producing results. Input consistency does not establish the provenance of the original simulation or training.
