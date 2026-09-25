# Additional implementation details

These details describe the supplied PET code. They supplement the [data map](data.md), [physics reference](physics.md) and [density model introduction](../learn/model.md); they do not establish detector performance or prescribe a new simulation run.

## Photon counting and the SiPM approximation

The simulation records optical-photon steps entering the continuous SiPM volume. Analysis counts accepted entries by `(EventID, ix, iy)` after assigning virtual pixels and rejecting gaps. It reads event identifiers and positions, without using `TrackID` to deduplicate. A photon that re-enters the readout could therefore contribute more than one entry; the stored count is not guaranteed to count unique photons.

For each event/channel, analysis samples `N_PE ~ Binomial(N_entries, 0.30)` using a seeded random generator. The response model has no wavelength-dependent detection probability, dark counts, crosstalk, afterpulsing, saturation, recovery or electronic noise. These omissions matter when comparing simulated count maps with real SiPM measurements.

Sources: `project/hpc-15-07/module-sim-doi-511-cone-array.py` (the `sipm_photons` actor); `project/hpc-15-07/analyse-array.py::build_npe_df`.

## Channel schemas across jobs

`build_doi_df` pivots only channels observed in its input. It fills missing event/channel combinations with zero, but does not explicitly create all 64 channel columns when a channel is absent from the entire job. Small jobs can consequently produce different headers.

The merge tool copies CSV rows under the first file's header. Its default header check rejects differences. `--allow-header-mismatch` merely allows copying to continue; it does not reorder columns or fill missing channels. Establish a common schema before merging. Event identifiers remain local to each job, so preserve job and seed identifiers separately.

Sources: `project/hpc-15-07/analyse-array.py::build_doi_df`; `project/hpc-15-07/merge_csvs_fast.py::merge_csvs`.

## The combined uncertainty selection

The trade-off diagnostic uses the predicted standard deviations for the requested spatial axes, normally x, y and z. Eligible rows have finite residuals and finite, positive standard deviations on every selected axis. For those rows it defines:

<div class="equation">s<sub>a</sub> = max[median(σ<sub>a</sub>), LOSS_EPS]<br>q = √[∑<sub>a</sub>(σ<sub>a</sub> / s<sub>a</sub>)²]</div>

The code scans distinct thresholds at the 5th through 95th percentiles of q, in five-percentile increments, and selects `q <= threshold`. The figure's label uses a strict inequality, but the implementation includes equality. Its selected fraction is relative to the eligible rows; it is not absolute PET sensitivity.

For each selected subset, the curve labelled “residual FWHM” is `2.355 × sample standard deviation`, using `ddof=1`. It is a Gaussian-equivalent width, not a fitted histogram FWHM. Subsets below the default minimum of 20 events receive no finite width. The median scales are estimated on the evaluated sample; comparing a fixed operational cut across datasets would require independently fixed scales.

Source: `project/hpc-15-07/evaluate_cnn_performance.py::plot_uncertainty_resolution_sensitivity_tradeoff`.

## The incoming scatter-energy surrogate

For incoming scattered-photon energy E′, the source code evaluates relative discrete weights using E₀ = 511 keV:

<div class="equation">r = E′ / E₀<br>cos θ = 2 − E₀ / E′<br>w(E′) = r + 1/r − sin² θ</div>

The implementation samples equally spaced energies between the requested limits, clipped inside `(E₀/3, E₀)` by 0.001 keV. Defaults are 48 energies between 180 and 505 keV. It clips cos θ to [−1, 1] and weights to nonnegative values, then supplies those weights to the discrete-spectrum source. They represent the implemented transformed Klein–Nishina shape, not measured patient-scatter data.

The scattered source retains the same spatial and cone-direction configuration as the primary source. The formula's θ does not impose a correlation on the emitted direction. Separate source counts are set by rounding the requested scattered fraction times the event count; no patient or annihilation pair is transported in this route.

Source: `project/hpc-15-07/module-sim-doi-511-cone-array.py::_patient_scatter_spectrum` and source construction.

## The conditional likelihood

For one event, let a be the true x or y coordinate, with predicted mean μ<sub>a</sub> and width σ<sub>a</sub>. Let π<sub>k</sub>, μ<sub>k</sub> and σ<sub>k</sub> be the z-mixture parameters. Omitting parameter-independent Gaussian constants, the code's loss is:

<div class="equation">ℓ = ∑<sub>a∈{x,y}</sub> [(a − μ<sub>a</sub>)² / (2σ<sub>a</sub>²) + log σ<sub>a</sub>]<br>− log {∑<sub>k</sub> [π<sub>k</sub> / σ<sub>k</sub>] exp[−(z − μ<sub>k</sub>)² / (2σ<sub>k</sub>²)]}</div>

The reported NLL averages ℓ over the batch and divides by three coordinates. The implementation uses log-softmax and log-sum-exp for the mixture term. Removing the Gaussian constants means a negative reported NLL is not by itself an error.

An optional soft bounds penalty is added separately to this NLL. It acts on the reported coordinate means; its detached normalization uses the mean three-dimensional position error. Its default relative weight is zero. It neither truncates the density at the crystal boundaries nor introduces correlations between coordinates.

Sources: `project/hpc-15-07/train-cnn-array-xyz-gmm.py::xy_zgmm_nll_terms`, `oob_penalty_terms_xyz` and `train_cnn`.

## Files produced by training

These are outputs defined by the trainer; the supplied examples do not include every file from the original trained models. Training and new-input inference remain [later extensions](../later.md).

| Output | Contents and interpretation |
| --- | --- |
| `cnn_xyz_best.pt` | PyTorch state dictionary selected by lowest validation total loss; architecture, preprocessing and all configuration are not embedded |
| `training-history.csv` | Epoch, training/validation NLL, mean 3D position error, weighted bounds penalty and total loss |
| `training-summary.json` | Best validation total loss, minimum validation 3D error, coordinate bounds, bounds-penalty weights, PE window, width bounds, K and output filenames |
| `test-predictions-with-uncertainty.csv` | Retained test metadata, xyz truth/predictions/widths, mixture mean and each component's weight, mean and width, evaluated using the selected checkpoint |
| `training-curves.png` | Training/validation NLL and total loss by epoch |
| `training-curves-3d-error.png` | Training/validation mean 3D position error by epoch |

`best_val_mean_3d_error_mm` is the minimum over epochs and can come from a different epoch than the saved checkpoint. The summary omits the random seed, normalization, exact input identity and full command. Record these separately with the source version, model K and data-selection history before relying on a run for reproducibility or inference.

Source: `project/hpc-15-07/train-cnn-array-xyz-gmm.py::train_cnn` and `main`.
