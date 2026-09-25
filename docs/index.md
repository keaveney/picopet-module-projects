<div class="eyebrow">UCD · PET detector research · Shared introduction</div>

# Find the interaction.<br>Understand the uncertainty.

<p class="intro">A gamma ray leaves a pattern of light. Learn how that pattern can tell us where it interacted—and how precisely we know.</p>

This guide introduces **picoPET module simulation and “3D chess”**, the CNN used to infer a three-dimensional interaction position from an 8 × 8 light map. It is written for final-year physics students joining Dr James Keaveney’s research project.

**Start with existing data.** You will inspect feature CSVs, evaluate saved CNN predictions, and interpret the resulting plots using the research scripts. A CPU is sufficient for these compact exercises; training, new simulation and cluster access come later.

<div class="route-grid" markdown>
<div class="route-card" markdown>
<span class="step">01 / UNDERSTAND</span>
### Meet the detector
Connect scintillation, light sharing and depth of interaction to the measured 64-channel map.

[Read the detector introduction →](learn/detector.md)
</div>
<div class="route-card" markdown>
<span class="step">02 / GET A FIRST RESULT</span>
### Work with a real CSV
Set up the local environment, check the input data and generate the first validation gallery.

[Begin local setup →](practicals/setup.md)
</div>
<div class="route-card" markdown>
<span class="step">03 / INTERPRET</span>
### Evaluate saved predictions
Distinguish residual bias, spread and uncertainty. Compare ways to summarise the same predicted density.

[Explore saved predictions →](practicals/predictions.md)
</div>
<div class="route-card" markdown>
<span class="step">04 / INVESTIGATE</span>
### Explore simulation and analysis
Complete the common route, then learn what is needed to generate new simulated data and analyse the module response.

[Explore later workflows →](later.md)
</div>
</div>

## What you should be able to explain

- How an interaction produces a light map, and why the map may contain depth information.
- What the simulation truth label actually records.
- How a conditional density differs from a single predicted position.
- Why a narrow fitted residual core is only one part of performance.
- What is traded away when selecting events with small predicted uncertainty.

## Your first session

Read [detector and light sharing](learn/detector.md) and [coordinates and truth](learn/coordinates.md). Follow [setup](practicals/setup.md), [read a light map](practicals/features.md), then [validate the distributions](practicals/validation.md). Keep a short notebook recording the command, input identity, output folder and your interpretation of one plot.

!!! note "A successful procedure is not a certified physics result"
    The supplied small datasets are independent historical examples with incomplete producing configurations. They are useful for checking procedures and learning interpretation. They do not establish production detector performance, PET image resolution or a validated clinical system.

## Research scope

All practical calculations use the research scripts in `project/hpc-15-07/`. The [example data checks](reference/data-checks.md) describe the teaching inputs and their limitations. The [project status](reference/status.md) lists open questions for the student investigations.

Distinguish what the code implements, what a particular dataset shows and what remains a hypothesis. Training, new-input inference and cluster simulation require additional preparation; see [later extensions](later.md).
