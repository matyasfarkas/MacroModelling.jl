#!/bin/bash
#
# Monitor smoke test, report results, and launch 1000-draw production run
#

set -e

REPO_ROOT="/Volumes/MacMini/matyasfarkas/Documents/GitHub/SurrogateNN_Estimation.jl"
cd "$REPO_ROOT"

SMOKE_PID=$1
if [ -z "$SMOKE_PID" ]; then
    echo "Usage: monitor_and_launch_1000.sh <smoke_test_pid>"
    exit 1
fi

echo "Monitoring smoke test PID $SMOKE_PID..."

# Wait for smoke test to finish
while kill -0 "$SMOKE_PID" 2>/dev/null; do
    sleep 30
done

echo ""
echo "========================================"
echo "SMOKE TEST FINISHED"
echo "$(date)"
echo "========================================"
echo ""

# Show the results
SMOKE_LOG=".local_artifacts/logs/linear_hmc_smoke.log"
if [ -f "$SMOKE_LOG" ]; then
    echo "=== SMOKE TEST RESULTS ==="
    # Extract non-ANSI lines from the end of the log
    grep -v '\[K\|\[A' "$SMOKE_LOG" | tail -80
    echo ""
fi

# Check if smoke test succeeded (output file exists)
SMOKE_OUT=".local_artifacts/hlt_18param_realdata/hlt_linear_hmc_advhmc_smoke.jls"
if [ -f "$SMOKE_OUT" ]; then
    echo "Smoke test output found: $SMOKE_OUT"
    echo "Launching 1000-draw production run..."
    echo ""
    nohup bash scripts/run_linear_hmc_1000.sh > .local_artifacts/logs/linear_hmc_1000.log 2>&1 &
    echo "1000-draw run launched as PID $!"
    echo "Log: .local_artifacts/logs/linear_hmc_1000.log"
else
    echo "WARNING: Smoke test output NOT found at $SMOKE_OUT"
    echo "The smoke test may have failed. Check the log above."
    echo "NOT launching 1000-draw run."
fi
