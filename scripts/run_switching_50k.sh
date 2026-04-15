#!/bin/bash
#
# 50K regime-switching estimation for paper results
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

SURROGATE=".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_rom1_v3.jls"
DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
GATE=".local_artifacts/hlt_18param_realdata/gate_calibration.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_switching_inversion_chain_50k.jls"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "50K REGIME-SWITCHING ESTIMATION"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
    "$SURROGATE" \
    "$DATA" \
    --gate-calibration="$GATE" \
    --out="$OUT" \
    --samples=50000 \
    --chains=1 \
    --sampler=mh \
    --shock-filter=inversion \
    --linear-filter=kalman \
    --obs-sigma-mode=max \
    --obs-sigma-scale=2.0 \
    --obs-sigma-floor=0.1 \
    --inversion-maxit=10 \
    --inversion-tol=1e-6 \
    --inversion-lambda=1e-4 \
    --gate-mode=soft \
    --gate-k-pre=4 \
    --gate-k-post=8 \
    --gate-min-len=4 \
    --gate-use-cached-stats \
    --param-set=phase1_18params \
    --seed=42 \
    --chunk-size=2500

echo ""
echo "==============================================="
echo "ESTIMATION COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
