#!/bin/bash
#
# Overnight regime-switching estimation with inversion filter
#
# This script runs the Smets-Wouters HLT model estimation using:
# 1. ROM1 (linear) for shock recovery via inversion filter
# 2. ROM1 + NN surrogate for likelihood evaluation
# 3. Kalman filter for linear (non-gate) periods
# 4. Regime-switching gate to select nonlinear vs linear periods
# 5. MH random walk sampling
#
# Key fix: uses --shock-filter=inversion instead of --shock-filter=sampling
# The inversion filter recovers shocks period-by-period, preventing state divergence
# that caused the previous -10^13 log-likelihood catastrophe.
#
# Usage:
#   cd /Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl
#   nohup bash scripts/run_overnight_switching_estimation.sh > .local_artifacts/logs/switching_estimation_overnight.log 2>&1 &

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

# Paths
SURROGATE=".local_artifacts/hlt_18param_validation_v2_combined/hlt_sep_surrogate_trained_rom1_v3.jls"
DATA=".local_artifacts/hlt_18param_realdata/hlt_real_data_payload.jls"
GATE=".local_artifacts/hlt_18param_realdata/gate_calibration.jls"
OUT=".local_artifacts/hlt_18param_realdata/hlt_switching_inversion_chain_overnight.jls"

# Create log directory
mkdir -p .local_artifacts/logs

echo "==============================================="
echo "OVERNIGHT REGIME-SWITCHING ESTIMATION"
echo "Started: $(date)"
echo "==============================================="
echo ""
echo "Surrogate: $SURROGATE"
echo "Data:      $DATA"
echo "Gate:      $GATE"
echo "Output:    $OUT"
echo ""

julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
    "$SURROGATE" \
    "$DATA" \
    --gate-calibration="$GATE" \
    --out="$OUT" \
    --samples=10000 \
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
    --chunk-size=500

echo ""
echo "==============================================="
echo "ESTIMATION COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
