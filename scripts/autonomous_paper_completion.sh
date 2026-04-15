#!/bin/bash
set -e  # Exit on first error

# Configuration
REPO_ROOT="/Users/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
LOG_DIR="$REPO_ROOT/.local_artifacts/logs"
OUTPUT_DIR="$REPO_ROOT/.local_artifacts/hlt_18param_validation_v2"

mkdir -p "$LOG_DIR"
mkdir -p "$OUTPUT_DIR"

cd "$REPO_ROOT"

# Phase 1: Environment
echo "[$(date)] Phase 1: Environment setup"
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()' \
  2>&1 | tee "$LOG_DIR/phase1_env_$(date +%Y%m%d_%H%M%S).log"

# Phase 2.1: Dataset generation
echo "[$(date)] Phase 2.1: Dataset generation (6 hours)"
julia --project=. scripts/hlt_sep_surrogate_dataset_generate.jl \
  --param-set phase1_18params_narrow \
  --theta-samples 125 \
  --samples-per-theta 184 \
  --sample-length 184 \
  --burn-in 100 \
  --sep-expectation-method hmc \
  --hmc-samples 200 \
  --hmc-warmup 100 \
  --hmc-leapfrog-steps 15 \
  --sep-horizon 20 \
  --sep-order 1 \
  --sep-nnodes 3 \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/phase2_1_dataset_$(date +%Y%m%d_%H%M%S).log"

# Phase 2.2-2.4: Train + Gate + Synthetic
echo "[$(date)] Phase 2.2: Surrogate training (5 min)"
julia --project=. scripts/hlt_sep_surrogate_train.jl \
  "$OUTPUT_DIR/dataset.jls" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/phase2_2_train_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Phase 2.3: Gate calibration (2 min)"
julia --project=. scripts/hlt_sep_surrogate_gate_calibration.jl \
  "$OUTPUT_DIR/payload.jls" \
  "$OUTPUT_DIR/surrogate_trained.jls" \
  --output-dir "$OUTPUT_DIR" \
  2>&1 | tee "$LOG_DIR/phase2_3_gate_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] Phase 2.4: Synthetic data (6 min)"
julia --project=. scripts/hlt_sep_surrogate_synthetic_data.jl \
  "$OUTPUT_DIR/payload.jls" \
  --output-dir "$OUTPUT_DIR/synthetic" \
  --sample-length 184 \
  --burn-in 100 \
  2>&1 | tee "$LOG_DIR/phase2_4_synthetic_$(date +%Y%m%d_%H%M%S).log"

# Phase 2.5 & 3: MCMC + Benchmarks (PARALLEL)
echo "[$(date)] Phase 2.5 + 3: MCMC estimation (20 hrs) + Benchmarks (8 hrs) in parallel"

# Start MCMC in background
julia --project=. scripts/hlt_sep_surrogate_synthetic_estimation.jl \
  "$OUTPUT_DIR/dataset.jls" \
  "$OUTPUT_DIR/payload.jls" \
  "$OUTPUT_DIR/synthetic/synthetic_data.jls" \
  --samples 2000 \
  --chains 4 \
  --sampler mh \
  --output-dir "$OUTPUT_DIR/estimation" \
  2>&1 | tee "$LOG_DIR/phase2_5_mcmc_$(date +%Y%m%d_%H%M%S).log" &

MCMC_PID=$!

# Start benchmarks in parallel
julia --project=. scripts/hlt_sep_surrogate_fom_benchmark.jl \
  "$OUTPUT_DIR/dataset.jls" \
  "$OUTPUT_DIR/payload.jls" \
  --test-size 100 \
  --methods rom_kalman,regime_switching,particle_filter,direct_sep \
  --output-dir "$REPO_ROOT/.local_artifacts/benchmarks" \
  2>&1 | tee "$LOG_DIR/phase3_benchmark_$(date +%Y%m%d_%H%M%S).log" &

BENCH_PID=$!

# Wait for both to complete
wait $MCMC_PID
echo "[$(date)] MCMC estimation complete"

wait $BENCH_PID
echo "[$(date)] Benchmarks complete"

# Phase 2.6: Table extraction
echo "[$(date)] Phase 2.6: Table extraction (2 min)"
julia --project=. scripts/extract_validation_tables.jl \
  "$OUTPUT_DIR/estimation" \
  --output-dir "$REPO_ROOT/docs/paper/generated" \
  --prefix 18param \
  2>&1 | tee "$LOG_DIR/phase2_6_extract_$(date +%Y%m%d_%H%M%S).log"

# Phase 4: Shock recovery figure
echo "[$(date)] Phase 4: Shock recovery figure (1 hour)"
if [ -f "$REPO_ROOT/scripts/extract_shock_recovery_figure.jl" ]; then
  julia --project=. scripts/extract_shock_recovery_figure.jl \
    "$OUTPUT_DIR/synthetic/synthetic_data.jls" \
    "$OUTPUT_DIR/estimation/chain.jls" \
    --output-dir "$REPO_ROOT/docs/paper/figures" \
    2>&1 | tee "$LOG_DIR/phase4_figures_$(date +%Y%m%d_%H%M%S).log"
else
  echo "WARNING: scripts/extract_shock_recovery_figure.jl not found. Skipping figure generation."
fi

# Phase 6: Final compilation
echo "[$(date)] Phase 6: Paper compilation (5 min)"
cd "$REPO_ROOT/docs/paper"
pdflatex -interaction=nonstopmode farkas_jmp_2026.tex
bibtex farkas_jmp_2026
pdflatex -interaction=nonstopmode farkas_jmp_2026.tex
pdflatex -interaction=nonstopmode farkas_jmp_2026.tex

# Verification
if grep -q "TBA" farkas_jmp_2026.pdf; then
  echo "WARNING: TBA still present in PDF"
  exit 1
else
  echo "SUCCESS: Paper complete, no TBA found"
  pdfinfo farkas_jmp_2026.pdf | grep Pages
  echo "Output: $REPO_ROOT/docs/paper/farkas_jmp_2026.pdf"
  exit 0
fi
