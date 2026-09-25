# Geometry and model details

These are **implemented defaults**, not independently measured detector specifications. The source of record is `project/hpc-15-07/`; candidate optical XMLs are in `project/hpc/`.

## Geometry and source

| Quantity | Current implementation |
| --- | --- |
| Crystals | 8 × 8; 3 × 3 × 15 mm; 3.2 mm pitch |
| Crystal-array transverse outer span | 25.4 mm (edges ±12.7 mm) |
| Glass / readout width | 25.8 mm |
| Crystal z bounds | −7.5 to +7.5 mm |
| Front glass | 4 mm; −11.5 to −7.5 mm |
| Rear glass | 0.1 mm; +7.5 to +7.6 mm |
| Silicon readout block | 0.5 mm; +7.6 to +8.1 mm |
| Source | Uniform spherical region, radius 13 mm, centre z = −200 mm |
| Cone half-angle | Approximately 2.22°, from `atan[0.3 × (25.8/2 + 13)/200]` |
| World | 1 m air cube |
| Effective EM physics name | `G4EmStandardPhysics` (the last assignment wins) |
| Production cuts | Gamma/electron: 0.1 mm |
| Optical kill condition | Crystal optical photons at global time ≥ 100 ns |

There are air-separated crystal daughters, no explicit segmented ESR walls, and no separate front foil solid. Ordered border surfaces provide the reflector approximation. The 100 ns cut uses global time, not each optical track’s age.

The default source uses a 511 keV line plus a discrete scatter surrogate: fraction 0.35, 48 energy values from 180 to 505 keV. Counts are rounded per source. `gamma_cone` selects monoenergetic gammas. There is no simulated patient, retained scatter energy-angle correlation, fluorine-18 decay or natural lutetium activity in this current route.

## Optical parameters and caveats

| Parameter | Candidate XML value | Interpretation |
| --- | --- | --- |
| Scintillation yield | 32,000 photons/MeV | Model input |
| Time constant | 36 ns | Model input |
| `RESOLUTIONSCALE` | 5.02 | Model input requiring supporting rationale |
| Crystal refractive index | About 1.80–1.85, wavelength dependent | Tabulated model input |
| Crystal optical absorption length | 80 mm | Distinct from gamma attenuation length |
| Glass index / absorption length | 1.5 / 10,000 mm | Model input |
| Silicon index / absorption length | 3.5 / 0.01 mm | Model input |
| Reflector | `unified`, `dielectric_metal`, `polished`, reflectivity 0.98 | Surface approximation |
| Photon detection probability | 0.30 in analysis | Binomial sampling; inactive `Detector_fast` does not set it |

The material called LYSO uses Lu₂SiO₅ stoichiometry without explicit yttrium substitution or cerium doping. `Diffuse_front` is polished with no explicit bulk-scattering model. An unused ESR entry contains the invalid number `0.0000001/`; no scientific input has been silently repaired.

## Attenuation model

For constant gamma attenuation coefficient μ, normal thickness L and incident angle θ, the conditional interaction-depth density is a normalized truncated exponential:

<div class="equation">p(d | θ, interaction) = (μ / cos θ) exp(−μd / cos θ) / [1 − exp(−μL / cos θ)]<br>0 ≤ d ≤ L; &nbsp; λ = 1 / μ</div>

The evaluator averages over 96 midpoint samples uniform in cos θ within the configured cone and fits μ to binned counts. It does not use an empirically reweighted incident-angle distribution. The approximation omits lateral escape and selection-dependent angular weighting. A mixture over angles is not exactly a single exponential at a mean coefficient.

The fit normally requests a 511 ± 1 keV selected-step energy window, but relaxes it when fewer than 50 rows pass. Always inspect the output selection fields.

## Learning and density details

The default trainer uses no image normalization, Adam at 3 × 10⁻⁴, and a seeded 70/15/15 row split. Widths are clamped through log-width bounds corresponding by default to 0.05–50 mm. Component means are unbounded. The coordinates factorize conditionally; z is an untruncated Gaussian mixture.

Negative log likelihood combines x/y Gaussian terms with a log-sum-exp z-mixture term, averaged over events and three coordinates. An optional soft bounds penalty has coefficient zero by default; it is not an active hard constraint. The 64-unit hidden layer does not imply a hard maximum of 21 components.

These implementation facts do not establish that uncertainties are calibrated or that the density family captures all optical ambiguities.

See the [implementation notes](implementation-notes.md) for photon counting, uncertainty selections, the training likelihood and saved outputs.
