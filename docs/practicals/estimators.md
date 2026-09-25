# 5 · Compare point estimators

<span class="status">Mean, median and mode</span>

## Why am I doing this?

Mean, median and mode answer different questions about **the same saved conditional density**. Changing the point estimate can change the residual core and tails without changing the learned density.

## What do I run?

**Working directory: `project/hpc-15-07`**, analysis environment active.

```bash
python evaluate_cnn_performance.py  \
  --predictions-csv ../hpc/local-cnn-zgmm-plain-test/test-predictions-with-uncertainty.csv  \
  --outdir student-runs/estimators  \
  --no-per-pixel-plots  \
  --n-z-resolution-bins 4  \
  --n-pred-z-condition-bins 4  \
  --n-pred-z-condition-gmm-components 2  \
  --z-gmm-point-estimators mean median mode
```

## What should I see?

Under `student-runs/estimators/`, expect `z-gmm-mean/`, `z-gmm-median/` and `z-gmm-mode/`, each with plots and metrics, plus a root `index.html` and `metrics-z-gmm-point-estimator-comparison.json`. The root gallery provides access to the results; the numerical JSON supports direct comparisons.

This command does not train three models. The mean is a weighted average; the median uses CDF bisection; the mode is a grid maximum **restricted to the physical z interval**. The mode may differ from the unrestricted global maximum and from the mean of the largest-weight component.

## How do I know it worked?

Open the three metric files and compare `spatial_mae_mm`, `spatial_rmse_mm`, and `pull.z.fit_fwhm` together with `pull.z.fit_ok`. The x/y point estimates are unchanged by the z estimator choice for this saved model. Their all-event error metrics should agree. z uncertainty values remain mixture standard deviations for all three evaluations.

Record which estimator looks best under each metric; do not assume one should win all of them. A mode can sharpen a central peak while giving some events much larger errors.

## Common failures and questions

If mixture parameters are missing from a different CSV, the evaluator cannot necessarily reconstruct all three estimates. Check for the full `z_gmm_weight_k`, `z_gmm_mu_k`, `z_gmm_sigma_k` groups rather than relying on the filename.

Which estimate would you select if the objective penalised squared error? Absolute error? Why does “highest density” not mean “lowest average squared error”? How does an in-range grid affect interpretation near a crystal boundary?

<div class="next" markdown>Next: [compare saved component-count studies →](components.md)</div>
