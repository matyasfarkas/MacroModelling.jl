#!/bin/bash
#
# 500K LINEAR HMC BASELINE (NUTS + Kalman via AdvancedHMC.jl)
#
# Production run: 500,000 post-warmup draws with 2000 warmup.
# Uses finite-difference gradients — no ForwardDiff Dual overhead.
#
# At first order, OBC = non-OBC since max(1,R*) linearizes to R*.
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_chain_500k.jls"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "500K LINEAR HMC BASELINE (AdvancedHMC NUTS)"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/run_linear_hmc_advancedhmc.jl \
    --data="$DATA" \
    --out="$OUT" \
    --samples=500000 \
    --adapt=2000 \
    --target-accept=0.65 \
    --max-depth=8 \
    --fd-eps=1e-5 \
    --seed=42

echo ""
echo "==============================================="
echo "500K LINEAR HMC COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
