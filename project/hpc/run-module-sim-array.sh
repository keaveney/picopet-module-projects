#!/bin/bash

set -euo pipefail

LOCAL_RUN=0
if [[ "${1:-}" == "--local" ]]; then
    LOCAL_RUN=1
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "${LOCAL_RUN}" -eq 1 ]]; then
    cd "${SCRIPT_DIR}"
else
    source /home/USER/venvs/py312-opengate-env/bin/activate
    cd "${SLURM_SUBMIT_DIR}"
fi

THREADS="${SLURM_CPUS_PER_TASK:-10}"
if [[ "${LOCAL_RUN}" -eq 1 ]]; then
    THREADS="${LOCAL_THREADS:-1}"
fi
EVENTS_PER_THREAD=10000
JOB_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
if [[ "${LOCAL_RUN}" -eq 1 ]]; then
    JOB_INDEX="${LOCAL_JOB_INDEX:-0}"
    EVENTS_PER_THREAD="${LOCAL_EVENTS_PER_THREAD:-10}"
fi
SIM_SEED=$((12345 + JOB_INDEX))
ANALYSIS_SEED=$((54321 + JOB_INDEX))

if [[ "${LOCAL_RUN}" -eq 1 ]]; then
    OUTDIR="${LOCAL_OUTDIR:-${SCRIPT_DIR}/outdir}"
else
    OUTDIR="/scratch/USER/module-sim/job_${JOB_INDEX}"
fi
PLOTDIR="${OUTDIR}/analyse-plots"
mkdir -p "${OUTDIR}" "${PLOTDIR}"

echo "========================================"
echo "LOCAL_RUN           = ${LOCAL_RUN}"
echo "SLURM_JOB_ID        = ${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID = ${JOB_INDEX}"
echo "HOSTNAME            = $(hostname)"
echo "THREADS             = ${THREADS}"
echo "EVENTS_PER_THREAD   = ${EVENTS_PER_THREAD}"
echo "LOCAL_EVENTS_PER_THREAD   = ${LOCAL_EVENTS_PER_THREAD}"
echo "SIM_SEED            = ${SIM_SEED}"
echo "ANALYSIS_SEED       = ${ANALYSIS_SEED}"
echo "OUTDIR              = ${OUTDIR}"
echo "========================================"

python module-sim-doi-511-cone-array.py \
    --threads "${THREADS}" \
    --events "${EVENTS_PER_THREAD}" \
    --seed "${SIM_SEED}" \
    --output-dir "${OUTDIR}"

[[ -f "${OUTDIR}/sipm_hits.root" ]] || { echo "Missing sipm_hits.root"; exit 1; }
[[ -f "${OUTDIR}/gamma_steps.root" ]] || { echo "Missing gamma_steps.root"; exit 1; }

python analyse-array.py \
    --sipm-root "${OUTDIR}/sipm_hits.root" \
    --gamma-root "${OUTDIR}/gamma_steps.root" \
    --output-csv "${OUTDIR}/df_training.csv" \
    --plot-dir "${PLOTDIR}" \
    --rng-seed "${ANALYSIS_SEED}"

# -------------------------------------------------
# If CSV exists and is non-empty, delete ROOT files
# -------------------------------------------------
CSV_FILE="${OUTDIR}/df_training.csv"

if [[ -s "${CSV_FILE}" ]]; then
    echo "Analysis CSV created successfully: ${CSV_FILE}"

    if [[ "${LOCAL_RUN}" -eq 1 ]]; then
        echo "Deleting large ROOT files to save space..."
        echo "local run - keeping root files for debugging"
    else
        rm -f "${OUTDIR}/sipm_hits.root"
        rm -f "${OUTDIR}/gamma_steps.root"
        echo "ROOT cleanup complete."
    fi
else
    echo "CSV output missing or empty: ${CSV_FILE}"
    echo "Keeping ROOT files for debugging."
    exit 1
fi
