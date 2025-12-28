#!/bin/bash
while ps -p 61923 > /dev/null 2>&1; do
    elapsed=$(ps -p 61923 -o etime= | tr -d ' ')
    cpu=$(ps -p 61923 -o %cpu= | tr -d ' ')
    echo "[$(date +%H:%M:%S)] Still running - Elapsed: $elapsed, CPU: ${cpu}%"
    sleep 180  # Check every 3 minutes
done
echo "[$(date +%H:%M:%S)] Process completed!"
