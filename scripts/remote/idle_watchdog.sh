#!/usr/bin/env bash
# Idle GPU watchdog: terminates the cloud instance if GPU utilization stays below threshold
# Usage: nohup ./scripts/remote/idle_watchdog.sh 15 5 > watchdog.log 2>&1 &
set -euo pipefail

MAX_IDLE_MINUTES="${1:-15}"
UTIL_THRESHOLD_PCT="${2:-5}"
CHECK_INTERVAL_SEC=60
IDLE_COUNT=0

echo "[sml-watchdog] Monitoring GPU utilization every ${CHECK_INTERVAL_SEC}s (shutdown after ${MAX_IDLE_MINUTES}m < ${UTIL_THRESHOLD_PCT}%)."

while true; do
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | awk '{ if ($1 > max) max=$1 } END { print max+0 }')
    if [ "${UTIL}" -lt "${UTIL_THRESHOLD_PCT}" ]; then
        IDLE_COUNT=$((IDLE_COUNT + 1))
        echo "[sml-watchdog] GPU utilization ${UTIL}% < ${UTIL_THRESHOLD_PCT}% (${IDLE_COUNT}/${MAX_IDLE_MINUTES})"
        if [ "${IDLE_COUNT}" -ge "${MAX_IDLE_MINUTES}" ]; then
            echo "[sml-watchdog] Idle limit reached. Shutting down instance to prevent wasted GPU billing."
            if command -v runpodctl >/dev/null 2>&1 && [ -n "${RUNPOD_POD_ID:-}" ]; then
                runpodctl stop pod "${RUNPOD_POD_ID}"
            else
                sudo shutdown -h now || poweroff || kill -TERM 1
            fi
            exit 0
        fi
    else
        IDLE_COUNT=0
    fi
    sleep "${CHECK_INTERVAL_SEC}"
done
