#!/bin/bash
#
# 500K Kalman filter estimation for paper results (linear benchmark)
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "500K KALMAN FILTER ESTIMATION"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/hlt_kalman_estimation.jl \
    --data=.local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls \
    --out=.local_artifacts/hlt_18param_realdata/hlt_kalman_mh_chain_500k.jls \
    --n-samples=500000 \
    --n-chains=1 \
    --burn-in=5000 \
    --param-set=phase1_18params \
    --seed=42

echo ""
echo "==============================================="
echo "ESTIMATION COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
