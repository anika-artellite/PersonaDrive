#!/bin/bash
# Start CARLA standalone so you can see exactly what's happening on boot.
# Run this BEFORE scripts/eval/test_one_lb1_route.sh - that script will detect a CARLA
# already up on $PORT and skip its own boot step.
#
# Usage:
#   bash scripts/eval/start_carla.sh                # default port 20000, foreground
#   PORT=20000 TM_PORT=30000 bash scripts/eval/start_carla.sh
#   nohup bash scripts/eval/start_carla.sh > carla.out 2>&1 &        # background
#
# When you're done, kill it with:
#   pkill -9 CarlaUE4
set -euo pipefail

REPO="${PERSONADRIVE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
: "${CARLA_ROOT:=${REPO}/software/carla0915}"
: "${PORT:=20000}"
: "${TM_PORT:=30000}"
: "${LOG:=${REPO}/eval_results/_carla_runtime.log}"

mkdir -p "$(dirname "${LOG}")"

if [ ! -f "${CARLA_ROOT}/CarlaUE4.sh" ]; then
    echo "ERROR: ${CARLA_ROOT}/CarlaUE4.sh not found"; exit 1
fi

# Refuse to step on an existing instance unless asked.
if pgrep -af CarlaUE4-Linux-Shipping >/dev/null 2>&1; then
    echo "WARNING: CarlaUE4 already running:"
    pgrep -af CarlaUE4-Linux-Shipping | head
    echo ""
    echo "Kill it first:  pkill -9 CarlaUE4"
    exit 2
fi

# Sanity: if the port is already in use, refuse - the would-be CARLA RPC
# server would silently fail to bind.
if (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | awk '{print $4}' | grep -q ":${PORT}\$"; then
    echo "ERROR: port ${PORT} is already in use"; exit 3
fi

echo ">>> CARLA_ROOT = ${CARLA_ROOT}"
echo ">>> world port = ${PORT}    TM port = ${TM_PORT}"
echo ">>> log file   = ${LOG}"
echo ""
echo ">>> launching CarlaUE4.sh -RenderOffScreen ..."
"${CARLA_ROOT}/CarlaUE4.sh" \
    -RenderOffScreen \
    -nosound \
    -carla-rpc-port="${PORT}" \
    -carla-streaming-port=$((PORT + 1)) \
    > "${LOG}" 2>&1 &
CARLA_PID=$!
echo ">>> CARLA pid = ${CARLA_PID}"
echo ""

# Watchdog: if CARLA dies, stop waiting.
echo ">>> waiting for RPC server to come up (up to 180s) ..."
for i in $(seq 1 180); do
    if ! kill -0 ${CARLA_PID} 2>/dev/null; then
        echo "ERROR: CARLA process died after ${i}s. Tail of log:"
        tail -40 "${LOG}"
        exit 4
    fi
    if grep -q "RPC server listening" "${LOG}" 2>/dev/null; then
        echo ">>> 'RPC server listening' detected after ${i}s"
        break
    fi
    if python -c "import socket; s=socket.socket(); s.settimeout(0.3); s.connect(('127.0.0.1',${PORT}))" 2>/dev/null; then
        echo ">>> port ${PORT} is open after ${i}s"
        break
    fi
    if [ $((i % 10)) -eq 0 ]; then
        # Periodic progress beat with a peek at the log
        echo "    [${i}s] still waiting; last log line: $(tail -1 "${LOG}")"
    fi
    sleep 1
done

# Final check
if python -c "import socket; s=socket.socket(); s.settimeout(0.3); s.connect(('127.0.0.1',${PORT}))" 2>/dev/null; then
    echo ""
    echo ">>> CARLA is UP on port ${PORT} (pid ${CARLA_PID})"
    echo ">>> log: ${LOG}"
    echo ""
    echo "Now run the agent in another terminal (or this terminal once you've"
    echo "backgrounded this script):"
    echo ""
    echo "    PORT=${PORT} TM_PORT=${TM_PORT} bash scripts/eval/test_one_lb1_route.sh"
    echo ""
    echo "When done, stop CARLA with:  kill ${CARLA_PID}"
    echo ""
    # If we're in foreground, hand the user something to watch. tail-follow
    # blocks until SIGINT, which keeps CARLA alive in the foreground case.
    echo ">>> tailing log (Ctrl-C will stop the tail; CARLA keeps running):"
    tail -f "${LOG}"
else
    echo ""
    echo "ERROR: CARLA didn't open port ${PORT} within 180s. Last 40 lines of log:"
    tail -40 "${LOG}"
    echo ""
    echo "Kill the dead instance:  kill ${CARLA_PID}"
    exit 5
fi