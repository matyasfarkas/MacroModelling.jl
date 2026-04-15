#!/bin/bash
# ============================================================================
# MONTE CARLO COVERAGE STUDY — Launch Script
# ============================================================================
#
# Runs the Monte Carlo coverage validation for the linear Kalman + NUTS-HMC
# estimator on the SW07-HLT model (18 parameters).
#
# Usage:
#   ./scripts/run_monte_carlo_coverage.sh              # default: 50 reps
#   ./scripts/run_monte_carlo_coverage.sh --smoke       # smoke test: 3 reps
#   ./scripts/run_monte_carlo_coverage.sh --resume      # resume from checkpoint
#   N_REP=100 SAMPLES=500 ./scripts/run_monte_carlo_coverage.sh  # custom
#
# Environment variables (all optional, defaults shown):
#   N_REP=50          Number of Monte Carlo replications
#   T_OBS=184         Observation periods (quarters)
#   SAMPLES=300       NUTS post-warmup draws per replication
#   ADAPT=200         NUTS warmup draws per replication
#   SEED=2026         Base random seed
#   DGP=prior         DGP mode: "prior" or "baseline_perturb"
#   MAX_DEPTH=8       NUTS maximum tree depth
#   TARGET_ACCEPT=0.65  NUTS target acceptance rate
#   CI_LEVEL=0.90     Credible interval level for coverage
#   START_REP=1       First replication to run (for manual parallelism)
# ============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Defaults
N_REP="${N_REP:-50}"
T_OBS="${T_OBS:-184}"
SAMPLES="${SAMPLES:-300}"
ADAPT="${ADAPT:-200}"
SEED="${SEED:-2026}"
DGP="${DGP:-prior}"
MAX_DEPTH="${MAX_DEPTH:-8}"
TARGET_ACCEPT="${TARGET_ACCEPT:-0.65}"
CI_LEVEL="${CI_LEVEL:-0.90}"
START_REP="${START_REP:-1}"
OUT_DIR="${OUT_DIR:-.local_artifacts/monte_carlo_coverage}"

# Handle command-line flags
for arg in "$@"; do
    case "$arg" in
        --smoke)
            N_REP=3
            SAMPLES=50
            ADAPT=50
            MAX_DEPTH=6
            OUT_DIR=".local_artifacts/monte_carlo_coverage_smoke"
            echo "[SMOKE TEST MODE: 3 reps, 50 draws]"
            ;;
        --resume)
            echo "[RESUME MODE: continuing from checkpoint]"
            ;;
        --help|-h)
            echo "Usage: $0 [--smoke] [--resume]"
            echo ""
            echo "Environment variables:"
            echo "  N_REP=$N_REP  T_OBS=$T_OBS  SAMPLES=$SAMPLES  ADAPT=$ADAPT"
            echo "  SEED=$SEED  DGP=$DGP  MAX_DEPTH=$MAX_DEPTH"
            echo "  TARGET_ACCEPT=$TARGET_ACCEPT  CI_LEVEL=$CI_LEVEL"
            echo "  START_REP=$START_REP  OUT_DIR=$OUT_DIR"
            exit 0
            ;;
    esac
done

# Create output and log directories
mkdir -p "$OUT_DIR"
mkdir -p .local_artifacts/logs

LOG_FILE=".local_artifacts/logs/mc_coverage_$(date +%Y%m%d_%H%M%S).log"

echo "============================================================"
echo "MONTE CARLO COVERAGE STUDY"
echo "Started: $(date)"
echo "============================================================"
echo "  Replications:   $N_REP (starting at $START_REP)"
echo "  T_obs:          $T_OBS"
echo "  Samples/rep:    $SAMPLES (+ $ADAPT warmup)"
echo "  DGP mode:       $DGP"
echo "  Max tree depth: $MAX_DEPTH"
echo "  Target accept:  $TARGET_ACCEPT"
echo "  CI level:       $CI_LEVEL"
echo "  Seed:           $SEED"
echo "  Output:         $OUT_DIR"
echo "  Log:            $LOG_FILE"
echo "============================================================"

# Run the Julia script
julia --project=. scripts/monte_carlo_coverage.jl \
    --n-rep="$N_REP" \
    --T-obs="$T_OBS" \
    --samples="$SAMPLES" \
    --adapt="$ADAPT" \
    --seed="$SEED" \
    --dgp="$DGP" \
    --max-depth="$MAX_DEPTH" \
    --target-accept="$TARGET_ACCEPT" \
    --ci-level="$CI_LEVEL" \
    --start-rep="$START_REP" \
    --out-dir="$OUT_DIR" \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "============================================================"
echo "MONTE CARLO COVERAGE STUDY COMPLETE"
echo "Finished: $(date)"
echo "Results:  $OUT_DIR/"
echo "Log:      $LOG_FILE"
echo "============================================================"
