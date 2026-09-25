# Coordinates, energy and truth

Before reading a resolution plot, be clear about **what was measured, what was used as truth and which rows survived selection**.

## Signed z is not depth from the front face

The crystal centre is `z = 0`. The front and rear crystal faces are at −7.5 and +7.5 mm. Depth measured from the front face is:

<div class="equation">d = z + 7.5 mm &nbsp;&nbsp; (0 ≤ d ≤ 15 mm)</div>

Thus `z = −5 mm` means a depth of 2.5 mm. A raw z histogram and a front-face-depth plot have different horizontal coordinates even for the same events; in practice their selections can also differ.

Crystal centres follow `x = 3.2 ix − 11.2 mm`, and likewise for y, with indices 0–7. A channel `pe_ix_iy` is stored at image location **[7 − iy, ix]**. This flips the array row direction so larger y appears towards the top.

## Three energy quantities that must stay distinct

| Quantity | Meaning | Typical stored unit |
| --- | --- | --- |
| `gamma_incident_energy` | Gamma kinetic energy immediately before the selected step | MeV |
| `gamma_delta_ke` | Gamma kinetic energy before minus after that step | MeV |
| `gamma_edep` | Local energy deposit assigned to that gamma step | MeV |
| `total_pe` | Sum of the 64 modelled detected counts | Photoelectrons |
| Visible-energy plot | Linear rescaling of total PE so its chosen mode maps to 511 keV | keV |

**Energy carried away by a secondary is not necessarily deposited locally in the gamma step.** Neither `gamma_edep` nor `gamma_delta_ke` is automatically the total deposited energy of the complete event. A visible-energy axis is a response calibration assumption, not direct access to true deposited energy.

## What “truth” means in this implementation

`build_truth_df()` selects the earliest retained energy-depositing **primary-gamma step**, excluding transportation and `UserSpecialCut`, with a local-deposit threshold. It prefers the recorded post-step position. These selected coordinates become `x_gamma`, `y_gamma`, `z_gamma` and subsequently the saved `*_true` targets.

!!! warning "Do not silently call this the first physical interaction"
    A physical interaction can transfer energy to a tracked secondary while leaving very little local gamma-step deposit. It can then be excluded. The current target needs an audit against unfiltered gamma steps before an unconditional “first physical interaction” claim is justified.

## Which events are missing?

Even without an explicit PE window, the feature CSV excludes zero-signal events, unmatched truth and rows removed for missing metadata. A depth distribution in this CSV is therefore a distribution of **retained rows**, not all emitted gammas.

Event IDs identify events within a job. Different jobs can reuse them. Photon and truth records are joined **before** merging job CSVs; the merge does not create a globally unique identifier.

## Check your understanding

- Convert `z = +6 mm` to front-face depth.
- Why might a cut on total light change the observed depth distribution?
- Could two rows with the same `event` value come from different simulated events?

??? note "Reasoning checks"
    The depth is 13.5 mm. Light collection can depend on position, so a light threshold can preferentially retain particular depths. Event IDs can repeat between jobs; an event ID alone is insufficient provenance after merging.

<div class="next" markdown>Next: [from light maps to conditional densities →](model.md)</div>
