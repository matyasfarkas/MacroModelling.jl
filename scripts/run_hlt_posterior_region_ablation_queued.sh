#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="hlt_posterior_region_local10_40draws_scales012505_queued_20260609"
OUT_DIR=".local_artifacts/counterfactual_decomposition/${RUN_ID}"
LOG_DIR=".local_artifacts/counterfactual_decomposition/logs"
LOG_PATH="${LOG_DIR}/${RUN_ID}.log"
CHAIN_PATH=".local_artifacts/hlt_18param_realdata/hlt_surrogate_hmc_extended_18p_pooled_8000_20260609.jls"
JULIA_BIN="${JULIA_BIN:-julia}"

mkdir -p "$OUT_DIR" "$LOG_DIR"

{
  echo "HLT posterior-region local ablation"
  echo "Run id: ${RUN_ID}"
  echo "Started wrapper: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  echo "Chain: ${CHAIN_PATH}"
  echo
} >> "$LOG_PATH"

active_blocking_screens() {
  screen -ls 2>/dev/null | grep -E "hlt_ablation_full_20260607|hlt_local10_profile_20260609" || true
}

while [[ -n "$(active_blocking_screens)" ]]; do
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') waiting for current ablation screens to finish..." >> "$LOG_PATH"
  sleep 1800
done

FIXED_OVERRIDES="calfa=0.2,cg=0.18,chabb=0.67,clandaw=1.1,constebeta=0.3,constepinf=0.7,crdy=0.0,crpi=1.5,crr=0.73,cry=0.125,csadjcost=4.89,csigl=2.0,ctou=0.025,ctrend=0.4,curvw=8.31,czcap=0.431818,cfc=1.2"

{
  echo "$(date '+%Y-%m-%dT%H:%M:%S%z') starting posterior-region ablation command"
  echo "Fixed non-estimated overrides: ${FIXED_OVERRIDES}"
  echo
} >> "$LOG_PATH"

"$JULIA_BIN" \
  --project=. \
  scripts/hlt_counterfactual_decomposition.jl \
  --variant-suite=local10 \
  --chain="$CHAIN_PATH" \
  --n-thetas=40 \
  --sim-periods=4 \
  --burn-in=2 \
  --shock-scales=0.1,0.25,0.5 \
  --sep-horizon=4 \
  --sep-maxit=120 \
  --sep-tol=1e-5 \
  --sep-accept-tol=0.5 \
  --sep-retry=true \
  --sep-retry-horizons=4,6 \
  --sep-retry-maxit-multipliers=1,2 \
  --sep-recovery=false \
  --sep-fallback-solver=qr \
  --checkpoint-every=1 \
  --param-overrides="$FIXED_OVERRIDES" \
  --output-dir="$OUT_DIR" \
  >> "$LOG_PATH" 2>&1

echo "$(date '+%Y-%m-%dT%H:%M:%S%z') posterior-region ablation finished" >> "$LOG_PATH"
