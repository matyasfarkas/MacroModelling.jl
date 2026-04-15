#!/bin/bash
#
# 8-Hour Autonomous Pipeline: Switching Posterior Grid + Coverage Analysis
# =========================================================================
#
# Phase A (0-6h):   Grid simulation with switching posterior chain
#                   Extended shock scales: 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5
#                   10 thetas × 2 trajectories per (scale, theta) = 220 SEP solves
#
# Phase B (6-6.5h): Coverage analysis with switching posterior
#                   Recovers smoothed shocks at switching posterior mean
#                   Generates comparative figures
#
# Phase C (6.5-7h): Extended grid with LINEAR posterior at higher shock scales (1.0, 1.2, 1.5)
#                   Augments the original grid with scales beyond 0.8
#
# Phase D (7-8h):   Final comparative analysis combining all results
#
# Usage:
#   nohup bash scripts/run_switching_grid_pipeline.sh > /tmp/switching_grid_pipeline.log 2>&1 &
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

LOGFILE="/tmp/switching_grid_pipeline.log"
SWITCH_CHAIN=".local_artifacts/hlt_18param_realdata/hlt_switching_synthetic_chain_2000.jls"
LINEAR_CHAIN=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_chain_2000_seed99.jls"
SWITCH_GRID_DIR=".local_artifacts/shock_scale_grid_switching"
LINEAR_EXT_DIR=".local_artifacts/shock_scale_grid_linear_extended"

echo "==============================================="
echo "SWITCHING POSTERIOR GRID PIPELINE"
echo "Started: $(date)"
echo "==============================================="
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE A: Grid simulation with switching posterior
# ═══════════════════════════════════════════════════════════════════════
echo "========== PHASE A: Switching posterior grid (est. 6 hours) =========="
echo "Started: $(date)"
echo ""

mkdir -p "$SWITCH_GRID_DIR"

julia --project=. scripts/shock_scale_grid_simulation.jl \
    --chain="$SWITCH_CHAIN" \
    --output-dir="$SWITCH_GRID_DIR" \
    --shock-scales="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1.0,1.2,1.5" \
    --n-thetas=10 \
    --n-trajectories=2 \
    --sep-accept-tol=0.35 \
    --sep-maxit=80 \
    --sep-tol=1e-5 \
    --sep-horizon=10 \
    --burn-in=20 \
    --sim-periods=40 \
    --seed=43

echo ""
echo "Phase A complete: $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE B: Coverage analysis with switching posterior
# ═══════════════════════════════════════════════════════════════════════
echo "========== PHASE B: Switching posterior coverage analysis =========="
echo "Started: $(date)"
echo ""

julia --project=. scripts/analyze_shock_scale_coverage.jl \
    --grid-dir="$SWITCH_GRID_DIR" \
    --chain="$SWITCH_CHAIN" \
    --fig-dir="$REPO_ROOT/docs/paper/Figures" \
    --fig-prefix="switching_grid"

echo ""
echo "Phase B complete: $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE C: Extended grid with LINEAR posterior (scales beyond 0.8)
# ═══════════════════════════════════════════════════════════════════════
echo "========== PHASE C: Linear posterior extended grid (est. 1 hour) =========="
echo "Started: $(date)"
echo ""

mkdir -p "$LINEAR_EXT_DIR"

julia --project=. scripts/shock_scale_grid_simulation.jl \
    --chain="$LINEAR_CHAIN" \
    --output-dir="$LINEAR_EXT_DIR" \
    --shock-scales="1.0,1.2,1.5" \
    --n-thetas=10 \
    --n-trajectories=2 \
    --sep-accept-tol=0.35 \
    --sep-maxit=80 \
    --sep-tol=1e-5 \
    --sep-horizon=10 \
    --burn-in=20 \
    --sim-periods=40 \
    --seed=44

echo ""
echo "Phase C complete: $(date)"
echo ""

# ═══════════════════════════════════════════════════════════════════════
# PHASE D: Comparative analysis combining all results
# ═══════════════════════════════════════════════════════════════════════
echo "========== PHASE D: Comparative analysis =========="
echo "Started: $(date)"
echo ""

julia --project=. scripts/analyze_shock_scale_coverage.jl \
    --grid-dir="$LINEAR_EXT_DIR" \
    --chain="$LINEAR_CHAIN" \
    --fig-dir="$REPO_ROOT/docs/paper/Figures" \
    --fig-prefix="linear_ext_grid"

echo ""
echo "==============================================="
echo "PIPELINE COMPLETE"
echo "Finished: $(date)"
echo "==============================================="
echo ""
echo "Results:"
echo "  Switching grid:      $SWITCH_GRID_DIR/"
echo "  Linear extended:     $LINEAR_EXT_DIR/"
echo "  Figures:             docs/paper/Figures/shock_scale_*.pdf"
