#!/bin/bash
#
# SMOKE TEST: Linear HMC baseline via AdvancedHMC.jl (200 draws + 100 warmup)
# Quick test to verify NUTS converges before committing to 500K.
#
# Uses finite-difference gradients through the Kalman filter — no ForwardDiff.
# At first order, OBC = non-OBC since max(1,R*) linearizes to R*.
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_advhmc_smoke.jls"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "SMOKE TEST: Linear HMC (AdvancedHMC, NUTS, 200 draws)"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/run_linear_hmc_advancedhmc.jl \
    --data="$DATA" \
    --out="$OUT" \
    --samples=200 \
    --adapt=100 \
    --target-accept=0.65 \
    --max-depth=8 \
    --fd-eps=1e-5 \
    --seed=42 \
    --verbose

echo ""
echo "==============================================="
echo "SMOKE TEST COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
