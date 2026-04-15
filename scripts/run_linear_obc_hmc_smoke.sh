#!/bin/bash
#
# SMOKE TEST: Linear HMC baseline (NUTS + Kalman, 100 draws)
# Quick test to verify NUTS converges before committing to 500K.
#
# NOTE: At first order the OBC max() is linearized around the non-binding SS,
# so linear OBC ≡ linear non-OBC.  We use the non-OBC model to avoid the
# state-dimension mismatch between the OBC auxiliary variables (114 states)
# and the surrogate training space (73 states).  The Kalman likelihood is
# identical either way.
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

SURROGATE=".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_rom1_v3.jls"
DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
GATE=".local_artifacts/hlt_18param_realdata/gate_calibration.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_smoke.jls"

mkdir -p .local_artifacts/logs

echo "==============================================="
echo "SMOKE TEST: Linear HMC baseline (NUTS, 100 draws)"
echo "Started: $(date)"
echo "==============================================="

julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
    "$SURROGATE" \
    "$DATA" \
    --gate-calibration="$GATE" \
    --out="$OUT" \
    --samples=0 \
    --linear-samples=100 \
    --chains=1 \
    --sampler=nuts \
    --nuts-adapt=50 \
    --nuts-target-accept=0.65 \
    --nuts-max-depth=8 \
    --shock-filter=inversion \
    --linear-filter=kalman \
    --obs-sigma-mode=max \
    --obs-sigma-scale=2.0 \
    --obs-sigma-floor=0.1 \
    --inversion-maxit=10 \
    --inversion-tol=1e-6 \
    --inversion-lambda=1e-4 \
    --param-set=phase1_18params \
    --seed=42

echo ""
echo "==============================================="
echo "SMOKE TEST COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
