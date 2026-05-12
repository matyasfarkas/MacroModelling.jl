#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_FAST_TESTS="${RUN_FAST_TESTS:-1}"
RUN_PAPER="${RUN_PAPER:-1}"

if [[ "$RUN_FAST_TESTS" == "1" ]]; then
  julia --project=. test/test_regime_switching_api.jl
  julia --project=. test/test_hlt_validation_harness.jl
  julia --project=. test/test_hlt_acceptance_smoke.jl
  julia --project=. test/test_sep_inversion_filter_likelihood.jl
fi

if [[ "$RUN_PAPER" == "1" ]]; then
  bash docs/SurrogateNN_paper/compile.sh
fi

echo "Replication smoke checks completed."
