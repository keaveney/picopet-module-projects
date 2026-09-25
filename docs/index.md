# PET detector simulation and analysis

This website is intended for final-year undergraduate physics students working with Dr James Keaveney and Dr. Sean Cournane at University College Dublin on research projects related to the development of novel detector modules for medical Positron Emission Tomography (PET). It is a work in progress and will develop in parallel to the projects themselves.

The main objectives of the website is to provide the minimal intorduction to PET, the low-cost context and need for novel detector concepts before provide a technical guide to setting up, running, and analysing simulations of PET modules using GATE 10 such that student get up to speed quickly. Details of specific resarch projects will not be provided on this site.

The current practicals use supplied datasets and research scripts to introduce the analysis. Instructions for running new simulations and training models will be developed separately; the [later workflows](later.md) describe their requirements and current limitations.

## Contents

- [Detector model and light sharing](learn/detector.md): module geometry, scintillation and the distribution of detected light across 64 channels.
- [Coordinates and simulation truth](learn/coordinates.md): position and energy definitions, truth selection and retained events.
- [Local practicals](practicals/setup.md): Python environment setup, inspection of the supplied data and analysis of light-response distributions.
- [Position reconstruction](learn/model.md): estimating interaction position from an 8 × 8 light map using a convolutional neural network, and interpreting its predicted probability distributions.
- [Prediction analysis](practicals/predictions.md): residual bias and spread, uncertainty estimates, event selections and comparisons of estimators.

## Getting started

Read the detector and coordinate definitions, then complete these practicals in order:

1. [Set up the Python environment](practicals/setup.md).
2. [Inspect a light map and check the input data](practicals/features.md).
3. [Analyse the light-response and interaction-depth distributions](practicals/validation.md).

These exercises run on a CPU using the supplied CSV files. They do not require a GPU, cluster access or a new simulation. Record the input file, command, software environment and output directory for each calculation, together with your interpretation of the results.

## Scripts, data and limitations

The practicals use the scripts in `project/hpc-15-07/`. The [data and script map](reference/data.md) identifies the inputs and entry points; the [example data checks](reference/data-checks.md) describe expected values and fit limitations.

The supplied datasets are small examples from earlier work, with incomplete simulation and training configurations. They support learning the analysis procedures, but do not establish detector performance. Check event selections and fit-success flags before interpreting a result. Outstanding requirements are listed in [project status](reference/status.md).
