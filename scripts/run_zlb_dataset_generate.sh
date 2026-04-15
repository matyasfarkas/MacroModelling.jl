#!/bin/bash
#
# ZLB-binding dataset generation (shock_scale=0.4)
# Generates supplementary training data where ZLB actually binds
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

OUTDIR=".local_artifacts/hlt_18param_validation_v2_combined/zlb_binding"
mkdir -p "$OUTDIR"
mkdir -p .local_artifacts/logs

echo "==============================================="
echo "ZLB-BINDING DATASET GENERATION (shock_scale=0.4)"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
    --use-zlb \
    --shock-scale=0.4 \
    --sep-accept-tol=0.50 \
    --sep-maxit=120 \
    --sep-tol=1e-4 \
    --sep-horizon=10 \
    --sep-nnodes=3 \
    --param-set=phase1_18params_narrow \
    --theta-samples=120 \
    --sample-length=184 \
    --burn-in=100 \
    --retry-shock-scale-backoff=0.9 \
    --output-dir="$OUTDIR" \
    --min-total-samples=5000

echo ""
echo "==============================================="
echo "ZLB DATASET GENERATION COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
