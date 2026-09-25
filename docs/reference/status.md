# Project status and open questions

## Scope of the student guide

The guide explains the supplied PET detector implementation and provides local exercises using its research scripts. The compact examples support learning and input-consistency checks. They do not establish experimental validation, full-run reproducibility or PET image resolution.

Geometry defaults, truth selection and estimator definitions describe the implementation in `project/hpc-15-07/`. Explanations such as total internal reflection contributing to DOI bias are hypotheses to investigate, rather than isolated causes established by the examples.

## Research priorities

1. **Baseline data:** identify a small dataset with complete simulation, analysis and model provenance and an agreed expected result.
2. **Student environment:** test a fresh installation on the students’ operating systems; resolve unavailable pinned packages explicitly.
3. **Truth audit:** confirm the intended target and compare current labels against unfiltered primary-gamma interactions.
4. **Production comparisons:** recover settings, splits, seeds and input identities behind the n1/n4/n8/n16 metrics before making causal performance claims.
5. **Later simulation:** confirm optical XMLs, material assumptions, the UCD Sonic environment and submission procedure.

The shared CSV exercises can be studied while these questions are resolved. Their [data checks](data-checks.md) and [provenance limits](data.md) define what conclusions the examples support.

## References and attribution

Scientific descriptions are grounded in the supplied research code. The project bibliography still needs references for PET DOI, scintillator properties, SiPM response, optical surfaces, GATE/Geant4 and mixture-density networks, together with supporting rationale for model constants.

The older SiPM library README retains its attribution to Dr Jesus Pena Rodriguez, CBM / Bergische Universitat Wuppertal, May 2024.
