#!/bin/bash
#
# 1000-DRAW LINEAR HMC BASELINE (NUTS + Kalman via AdvancedHMC.jl)
#
# Production run: 1000 post-warmup draws with 500 warmup.
# Target: ESS ~600 per parameter (60% efficiency typical for well-tuned NUTS).
#
# Uses finite-difference gradients — no ForwardDiff Dual overhead.
# At first order, OBC = non-OBC since max(1,R*) linearizes to R*.
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_chain_1000.jls"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "1000-DRAW LINEAR HMC BASELINE (AdvancedHMC NUTS)"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/run_linear_hmc_advancedhmc.jl \
    --data="$DATA" \
    --out="$OUT" \
    --samples=1000 \
    --adapt=500 \
    --target-accept=0.65 \
    --max-depth=8 \
    --fd-eps=1e-5 \
    --seed=42

echo ""
echo "==============================================="
echo "1000-DRAW LINEAR HMC COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
