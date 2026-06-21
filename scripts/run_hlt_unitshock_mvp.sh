#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUN_ROOT="${RUN_ROOT:-.local_artifacts/hlt_18param_validation_unitshock_mvp_20260620}"
JULIA_BIN="${JULIA_BIN:-julia}"
MODE="${1:-}"

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

case "$MODE" in
  base)
    log_and_run "$RUN_ROOT/logs/base_dataset.log" \
      "$JULIA_BIN" --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
        --no-obc \
        --param-set=phase1_18params_narrow \
        --theta-sampling=lhs \
        --theta-samples=32 \
        --samples-per-theta=64 \
        --sample-length=96 \
        --burn-in=80 \
        --sep-horizon=12 \
        --sep-order=1 \
        --sep-nnodes=3 \
        --sep-maxit=80 \
        --sep-tol=1e-4 \
        --sep-accept-tol=0.35 \
        --shock-scaling=none \
        --shock-scale=0.08 \
        --rom-orders=1 \
        --checkpoint-every=1 \
        --progress-every=1 \
        --timing \
        --min-total-samples=1024 \
        --output-dir="$RUN_ROOT/base"
    ;;
  zlb)
    log_and_run "$RUN_ROOT/logs/zlb_dataset.log" \
      "$JULIA_BIN" --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
        --use-zlb \
        --param-set=phase1_18params_narrow \
        --theta-sampling=lhs \
        --theta-samples=28 \
        --samples-per-theta=64 \
        --sample-length=96 \
        --burn-in=80 \
        --sep-horizon=8 \
        --sep-order=1 \
        --sep-nnodes=3 \
        --sep-maxit=100 \
        --sep-tol=1e-4 \
        --sep-accept-tol=0.50 \
        --stable-prefix \
        --stable-min-periods=64 \
        --theta-attempts-per-theta=2 \
        --retry-on-early-failure=true \
        --retry-shock-scale-backoff=0.85 \
        --shock-scaling=none \
        --shock-scale=0.25 \
        --rom-orders=1 \
        --checkpoint-every=1 \
        --progress-every=1 \
        --timing \
        --min-total-samples=1024 \
        --output-dir="$RUN_ROOT/zlb_binding"
    ;;
  combine-train)
    base_path="$RUN_ROOT/base/hlt_sep_surrogate_dataset.jls"
    zlb_path="$RUN_ROOT/zlb_binding/hlt_sep_surrogate_dataset.jls"
    if [[ ! -f "$base_path" ]]; then
      echo "Missing completed base dataset: $base_path" >&2
      exit 1
    fi
    if [[ ! -f "$zlb_path" ]]; then
      zlb_path="$RUN_ROOT/zlb_binding/hlt_sep_surrogate_dataset_checkpoint.jls"
    fi
    if [[ ! -f "$zlb_path" ]]; then
      echo "Missing completed or checkpointed ZLB dataset: $zlb_path" >&2
      exit 1
    fi
    combined_path="$RUN_ROOT/hlt_sep_surrogate_dataset_combined_with_zlb_unitshock_mvp.jls"
    surrogate_path="$RUN_ROOT/hlt_sep_surrogate_trained_with_zlb_unitshock_mvp.jls"
    {
      echo "Started: $(date '+%Y-%m-%dT%H:%M:%S%z')"
      echo "Repo: $REPO_ROOT"
      echo "Run root: $RUN_ROOT"
      echo "Combining: $base_path + $zlb_path -> $combined_path"
      "$JULIA_BIN" --project=. scripts/combine_zlb_dataset.jl \
        --base="$base_path" \
        --zlb="$zlb_path" \
        --out="$combined_path"
      echo
      echo "Training surrogate: $combined_path -> $surrogate_path"
      "$JULIA_BIN" --project=. scripts/hlt_sep_surrogate_train.jl \
        "$combined_path" \
        --rom-residual=1 \
        --obs-only \
        --epochs=250 \
        --hidden=192 \
        --hidden2=96 \
        --batch=512 \
        --seed=1 \
        --out="$surrogate_path"
      echo
      echo "Finished: $(date '+%Y-%m-%dT%H:%M:%S%z')"
    } > "$RUN_ROOT/logs/combine_train.log" 2>&1
    ;;
  *)
    echo "Usage: RUN_ROOT=<dir> JULIA_BIN=<julia> $0 {base|zlb|combine-train}" >&2
    exit 2
    ;;
esac
