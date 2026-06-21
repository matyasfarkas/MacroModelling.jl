#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-.local_artifacts/hlt_18param_validation_unitshock_20260619}"
JULIA_BIN="${JULIA_BIN:-julia}"
MODE="${1:-}"

DATA_PATH="${DATA_PATH:-.local_artifacts/hlt_18param_realdata/hlt_real_data_payload_extended_18p_unitshock_nonobc_20260620.jls}"
GATE_PATH="${GATE_PATH:-.local_artifacts/hlt_18param_realdata/gate_calibration_extended_18p_unitshock_20260619.jls}"
INIT_FROM="${INIT_FROM:-.local_artifacts/hlt_18param_realdata/hlt_linear_hmc_extended_18p_2000.jls}"
SURROGATE_PATH="${SURROGATE_PATH:-$RUN_ROOT/hlt_sep_surrogate_trained_with_zlb_unitshock.jls}"

SMOKE_ADAPT="${SMOKE_ADAPT:-25}"
SMOKE_SAMPLES="${SMOKE_SAMPLES:-25}"
SMOKE_MAX_DEPTH="${SMOKE_MAX_DEPTH:-5}"
FULL_ADAPT="${FULL_ADAPT:-500}"
FULL_SAMPLES="${FULL_SAMPLES:-1000}"
FULL_MAX_DEPTH="${FULL_MAX_DEPTH:-8}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.80}"
HMC_SEED="${HMC_SEED:-42}"

mkdir -p "$RUN_ROOT/base" "$RUN_ROOT/zlb_binding" "$RUN_ROOT/logs"

log_and_run() {
  local log_path="$1"
  shift
  {
    echo "Started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    echo "Repo: $REPO_ROOT"
    echo "Run root: $RUN_ROOT"
    echo "Command: $*"
    echo
    "$@"
    echo
    echo "Finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
  } > "$log_path" 2>&1
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

run_hmc() {
  local label="$1"
  local disable_nn="$2"
  local adapt="$3"
  local samples="$4"
  local max_depth="$5"
  local seed="$6"
  local out_path="$RUN_ROOT/hmc_${label}.jls"
  local log_path="$RUN_ROOT/logs/hmc_${label}.log"

  require_file "$SURROGATE_PATH" "trained surrogate"
  require_file "$DATA_PATH" "real-data payload"
  require_file "$GATE_PATH" "gate calibration"
  require_file "$INIT_FROM" "initialization chain"

  log_and_run "$log_path" \
    "$JULIA_BIN" --project=. scripts/run_surrogate_hmc_advancedhmc.jl \
      --surrogate="$SURROGATE_PATH" \
      --data="$DATA_PATH" \
      --gate-calibration="$GATE_PATH" \
      --init-from="$INIT_FROM" \
      --out="$out_path" \
      --adapt="$adapt" \
      --samples="$samples" \
      --seed="$seed" \
      --target-accept="$TARGET_ACCEPT" \
      --max-depth="$max_depth" \
      --checkpoint-every=50 \
      --disable-nn-correction="$disable_nn"
}

case "$MODE" in
  base)
    log_and_run "$RUN_ROOT/logs/base_dataset.log" \
      "$JULIA_BIN" --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
        --no-obc \
        --param-set=phase1_18params_narrow \
        --theta-sampling=prior \
        --theta-samples=125 \
        --samples-per-theta=184 \
        --sample-length=184 \
        --burn-in=100 \
        --sep-horizon=20 \
        --sep-order=1 \
        --sep-nnodes=3 \
        --sep-maxit=100 \
        --sep-tol=1e-5 \
        --sep-accept-tol=0.35 \
        --shock-scaling=none \
        --shock-scale=0.1 \
        --rom-orders=1,2 \
        --checkpoint-every=1 \
        --progress-every=1 \
        --timing \
        --output-dir="$RUN_ROOT/base"
    ;;
  zlb)
    log_and_run "$RUN_ROOT/logs/zlb_dataset.log" \
      "$JULIA_BIN" --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
        --use-zlb \
        --shock-scaling=none \
        --shock-scale=0.4 \
        --sep-accept-tol=0.50 \
        --sep-maxit=120 \
        --sep-tol=1e-4 \
        --sep-horizon=10 \
        --sep-order=1 \
        --sep-nnodes=3 \
        --param-set=phase1_18params_narrow \
        --theta-sampling=prior \
        --theta-samples=120 \
        --sample-length=184 \
        --burn-in=100 \
        --stable-prefix \
        --stable-min-periods=128 \
        --theta-attempts-per-theta=2 \
        --retry-on-early-failure=true \
        --retry-shock-scale-backoff=0.9 \
        --min-total-samples=5000 \
        --rom-orders=1,2 \
        --checkpoint-every=1 \
        --progress-every=1 \
        --timing \
        --output-dir="$RUN_ROOT/zlb_binding"
    ;;
  combine-train)
    base_path="$RUN_ROOT/base/hlt_sep_surrogate_dataset.jls"
    zlb_path="$RUN_ROOT/zlb_binding/hlt_sep_surrogate_dataset.jls"
    require_file "$base_path" "completed base dataset"
    if [[ ! -f "$zlb_path" ]]; then
      zlb_path="$RUN_ROOT/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls"
    fi
    require_file "$zlb_path" "completed or checkpointed ZLB dataset"
    combined_path="$RUN_ROOT/hlt_sep_surrogate_dataset_combined_with_zlb_unitshock.jls"
    surrogate_path="$SURROGATE_PATH"
    log_and_run "$RUN_ROOT/logs/combine_train.log" \
      "$JULIA_BIN" --project=. scripts/combine_zlb_dataset.jl \
        --base="$base_path" \
        --zlb="$zlb_path" \
        --out="$combined_path"
    log_and_run "$RUN_ROOT/logs/train_surrogate.log" \
      "$JULIA_BIN" --project=. scripts/hlt_sep_surrogate_train.jl \
        "$combined_path" \
        --rom-residual=1 \
        --obs-only \
        --epochs=400 \
        --hidden=256 \
        --hidden2=128 \
        --seed=1 \
        --out="$surrogate_path"
    ;;
  hmc-smoke-full)
    run_hmc "smoke_fullnn_seed${HMC_SEED}" false "$SMOKE_ADAPT" "$SMOKE_SAMPLES" "$SMOKE_MAX_DEPTH" "$HMC_SEED"
    ;;
  hmc-smoke-lineargate)
    run_hmc "smoke_lineargate_seed${HMC_SEED}" true "$SMOKE_ADAPT" "$SMOKE_SAMPLES" "$SMOKE_MAX_DEPTH" "$HMC_SEED"
    ;;
  hmc-full)
    run_hmc "fullnn_${FULL_ADAPT}x${FULL_SAMPLES}_seed${HMC_SEED}" false "$FULL_ADAPT" "$FULL_SAMPLES" "$FULL_MAX_DEPTH" "$HMC_SEED"
    ;;
  hmc-lineargate)
    run_hmc "lineargate_${FULL_ADAPT}x${FULL_SAMPLES}_seed${HMC_SEED}" true "$FULL_ADAPT" "$FULL_SAMPLES" "$FULL_MAX_DEPTH" "$HMC_SEED"
    ;;
  *)
    echo "Usage: RUN_ROOT=<dir> JULIA_BIN=<julia> $0 {base|zlb|combine-train|hmc-smoke-full|hmc-smoke-lineargate|hmc-full|hmc-lineargate}" >&2
    echo "Optional HMC env vars: DATA_PATH=... GATE_PATH=... INIT_FROM=... SURROGATE_PATH=... HMC_SEED=... FULL_ADAPT=... FULL_SAMPLES=..." >&2
    exit 2
    ;;
esac
