#!/bin/bash
# kill_all_sweeps.sh - Kill all TTQ sweep tmux sessions

SESSION_PREFIX="ttq-sweep"

echo "========================================="
echo "Kill All TTQ Sweep Sessions"
echo "========================================="
echo ""

# Find all sweep sessions
SESSIONS=$(tmux list-sessions 2>/dev/null | grep "${SESSION_PREFIX}" | cut -d: -f1)

if [ -z "$SESSIONS" ]; then
    echo "No active sweep sessions found."
    exit 0
fi

echo "Found sessions:"
echo "$SESSIONS"
echo ""

read -p "Kill all these sessions? (y/N) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    for SESSION in $SESSIONS; do
        echo "Killing $SESSION..."
        tmux kill-session -t $SESSION
    done
    echo ""
    echo "✓ All sweep sessions killed."
else
    echo "Cancelled."
fi
