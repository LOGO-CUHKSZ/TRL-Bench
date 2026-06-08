#!/bin/bash
# ==============================================================================
# Tail Training Log - Find and follow the active training log with progress bars
# ==============================================================================
# Usage: bash scripts/utils/tail_training_log.sh [pattern]
#   pattern: Optional grep pattern to match log files (default: latest)
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LOG_DIR="multinode_logs"
PATTERN="${1:-}"

# Find the most recently modified log file (or matching pattern)
if [ -n "$PATTERN" ]; then
    LATEST_LOGS=$(ls -t "$LOG_DIR"/*${PATTERN}*.log 2>/dev/null)
else
    LATEST_LOGS=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -10)
fi

if [ -z "$LATEST_LOGS" ]; then
    echo "❌ No log files found in $LOG_DIR"
    exit 1
fi

# Get unique timestamps from latest logs
TIMESTAMP=$(echo "$LATEST_LOGS" | head -1 | grep -oP '\d{8}_\d{6}' | head -1)

if [ -z "$TIMESTAMP" ]; then
    echo "❌ Could not extract timestamp from log filename"
    exit 1
fi

echo "📋 Found training run from timestamp: $TIMESTAMP"
echo ""

# Find all logs for this training run
RUN_LOGS=("$LOG_DIR"/*"$TIMESTAMP"*.log)

if [ ${#RUN_LOGS[@]} -eq 0 ]; then
    echo "❌ No logs found for timestamp $TIMESTAMP"
    exit 1
fi

echo "📊 Analyzing ${#RUN_LOGS[@]} log files..."
echo ""

# Find the largest log (likely the one with progress bars)
LARGEST_LOG=""
LARGEST_SIZE=0

for log in "${RUN_LOGS[@]}"; do
    if [ -f "$log" ]; then
        SIZE=$(stat -c%s "$log" 2>/dev/null || stat -f%z "$log" 2>/dev/null || echo 0)
        NODE=$(basename "$log" | grep -oP 'node_\K\d+')

        # Check if this log has progress bars (contains "[A" ANSI escape code)
        HAS_PROGRESS=$(grep -q '\[A' "$log" 2>/dev/null && echo "YES" || echo "NO")

        printf "  Node %-2s: %6s KB  [Progress: %s]\n" "$NODE" "$((SIZE/1024))" "$HAS_PROGRESS"

        if [ "$HAS_PROGRESS" = "YES" ] && [ $SIZE -gt $LARGEST_SIZE ]; then
            LARGEST_LOG="$log"
            LARGEST_SIZE=$SIZE
        fi
    fi
done

echo ""

if [ -z "$LARGEST_LOG" ]; then
    # Fallback: use largest log even without progress bars
    for log in "${RUN_LOGS[@]}"; do
        if [ -f "$log" ]; then
            SIZE=$(stat -c%s "$log" 2>/dev/null || stat -f%z "$log" 2>/dev/null || echo 0)
            if [ $SIZE -gt $LARGEST_SIZE ]; then
                LARGEST_LOG="$log"
                LARGEST_SIZE=$SIZE
            fi
        fi
    done
fi

if [ -z "$LARGEST_LOG" ]; then
    echo "❌ Could not identify active training log"
    exit 1
fi

NODE_NUM=$(basename "$LARGEST_LOG" | grep -oP 'node_\K\d+')

echo "✅ Active training log identified: Node $NODE_NUM"
echo "📂 Log file: $LARGEST_LOG"
echo ""
echo "Following log (Ctrl+C to stop)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

tail -f "$LARGEST_LOG"
