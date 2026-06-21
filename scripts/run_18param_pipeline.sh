#!/bin/bash
# 18-Parameter Validation Pipeline (Phase 2: Steps 2-6)
# Run this after dataset generation (Step 1) completes successfully.
#
# Usage:
#   bash scripts/run_18param_pipeline.sh [BASE_DIR]
#
# Default BASE_DIR: .local_artifacts/hlt_18param_validation_v2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BASE_DIR="${1:-.local_artifacts/hlt_18param_validation_v2}"
LOG_DIR=".local_artifacts/logs"
DATASET="$BASE_DIR/hlt_sep_surrogate_dataset.jls"

mkdir -p "$LOG_DIR" "$BASE_DIR/synthetic" "$BASE_DIR/estimation"

echo "=== 18-Parameter Validation Pipeline ==="
echo "Base directory: $BASE_DIR"
echo "Dataset: $DATASET"
echo "Started: $(date)"

# Verify dataset exists
if [ ! -f "$DATASET" ]; then
    echo "ERROR: Dataset not found: $DATASET"
    echo "Run dataset generation first (Step 1)."
    exit 1
fi

echo ""
echo "=== Step 2: Surrogate Training ==="
echo "Started: $(date)"
julia --project=. scripts/hlt_sep_surrogate_train.jl \
    "$DATASET" \
    --epochs=800 --hidden=512 --hidden2=256 --seed=1 \
    --lr=5e-4 --weight-decay=1e-5 --clip-norm=5.0 \
    --rom-residual=1 \
    --out="$BASE_DIR/hlt_sep_surrogate_trained.jls" \
    2>&1 | tee "$LOG_DIR/train_18param.log"
echo "Completed: $(date)"

SURROGATE="$BASE_DIR/hlt_sep_surrogate_trained.jls"
if [ ! -f "$SURROGATE" ]; then
    echo "ERROR: Surrogate file not created: $SURROGATE"
    exit 1
fi

echo ""
echo "=== Step 3: Synthetic Data Generation ==="
echo "Started: $(date)"
julia --project=. scripts/hlt_sep_surrogate_synthetic_data.jl \
    --sample-length=184 --burn-in=100 \
    --shock-scaling=none --shock-scale=0.1 \
    --sep-horizon=20 --sep-order=1 --sep-nnodes=3 \
    --sep-maxit=200 --sep-tol=1e-5 \
    --sep-linear-solver=normal_equations \
    --sep-accept-tol=0.5 \
    --attempts=5 --retry-on-early-failure \
    --seed=42 \
    --output-dir="$BASE_DIR/synthetic" \
    2>&1 | tee "$LOG_DIR/synthetic_18param.log"
echo "Completed: $(date)"

SYNTH_DATA="$BASE_DIR/synthetic/hlt_sep_synth_data.jls"
if [ ! -f "$SYNTH_DATA" ]; then
    echo "ERROR: Synthetic data not created: $SYNTH_DATA"
    exit 1
fi

echo ""
echo "=== Step 4: Gate Calibration ==="
echo "Started: $(date)"
julia --project=. scripts/hlt_sep_surrogate_gate_calibration.jl \
    "$SYNTH_DATA" \
    --out="$BASE_DIR/synthetic/gate_calibration.jls" \
    2>&1 | tee "$LOG_DIR/gate_18param.log"
echo "Completed: $(date)"

echo ""
echo "=== Step 5: MCMC Estimation (this will take 18-24 hours) ==="
echo "Started: $(date)"
julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
    "$SURROGATE" \
    "$SYNTH_DATA" \
    --samples=2000 --chains=4 --sampler=mh \
    --output-dir="$BASE_DIR/estimation" \
    2>&1 | tee "$LOG_DIR/estimation_18param.log"
echo "Completed: $(date)"

echo ""
echo "=== Step 6: Table Extraction ==="
echo "Started: $(date)"
julia --project=. scripts/extract_validation_tables.jl \
    "$BASE_DIR" \
    --output-dir=docs/paper/generated --prefix=18param \
    2>&1 | tee "$LOG_DIR/extract_18param.log"
echo "Completed: $(date)"

echo ""
echo "=== Pipeline Complete ==="
echo "Finished: $(date)"
echo "Check docs/paper/generated/ for output tables."
