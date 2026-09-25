# Later extensions

The introductory practicals focus on understanding existing measurements and saved predictions. These later stages explain where new data and models would come from. They are not prerequisites for the common practicals.

## Inference on new images

Inference applies trained weights to new feature maps. The evaluator only reads existing outputs. Before documenting a runnable inference route, obtain:

- A matching `cnn_xyz_best.pt` checkpoint and the model’s K.
- The exact architecture, input normalization and feature schema.
- Model version, training data identity and selection history.
- A tested inference entry point and a small approved reference output.

The checkpoint format is a state dictionary; it does not embed all necessary configuration. **No checkpoint or tested standalone inference command is supplied.**

## Training a new density model

The authoritative trainer is `project/hpc-15-07/train-cnn-array-xyz-gmm.py`. Conceptually it builds light maps, splits rows into 70% training / 15% validation / 15% test, optimises the conditional likelihood and saves predictions from the selected checkpoint.

The checkpoint is selected by lowest validation total loss. The smallest recorded validation 3D error can occur at a different epoch. A row-based split also needs scrutiny if simulation jobs share seeds or provenance. Test mode still reads the full CSV and constructs images; it is not a streaming low-memory loader.

Before a training practical is added, agree the dataset, independent split, normalization, K, seeds, compute budget and scientific objective. Record the full command and data hashes.

## Simulation and reduction on UCD Sonic

The current GATE simulation constructs geometry, emits gammas and transports secondaries and optical photons. `analyse-array.py` normally runs in the **same cluster job** to assign virtual channels, sample photoelectrons, select truth and reduce ROOT data to CSV. This avoids transferring large ROOT outputs for the introductory local work.

The current default source combines a 511 keV line with a discrete incoming scatter-energy surrogate. It does not simulate a patient or a back-to-back annihilation pair. The energy-angle correlations of patient transport are not retained by that surrogate.

Executable cluster instructions need the authoritative optical XML files, the UCD Sonic environment, current submission settings, approved paths and resource limits. Candidate XML files are supplied in `project/hpc/`, but their production identity is unconfirmed. An inactive ESR numeric value is malformed. The old driver contains obsolete paths and is not a verified submission recipe.

## Merging and provenance

Per-job CSVs must have compatible headers. The merge tool concatenates rows; it does not realign mismatched columns, deduplicate event IDs or introduce job identifiers. Preserve source-job and seed records before scaling up.

**Status:** these extensions still need executable procedures checked in the intended research environment before they become student practicals.
