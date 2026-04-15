#!/bin/bash
#
# LINEAR HMC — Farkas-Tatar (2020) matching run
#
# Fixes cmap=0.80, cmaw=0.89 (SW07 ARMA markup structure)
# to compare against the Farkas-Tatar IMFS WP 144 results.
#
# 1000 post-warmup + 500 warmup draws.
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_ft_match.jls"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "LINEAR HMC — Farkas-Tatar (2020) Matching Run"
echo "cmap=0.80, cmaw=0.89 (SW07 ARMA markups)"
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
    --seed=42 \
    --cmap=0.80 \
    --cmaw=0.89

echo ""
echo "==============================================="
echo "FARKAS-TATAR MATCHING RUN COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
