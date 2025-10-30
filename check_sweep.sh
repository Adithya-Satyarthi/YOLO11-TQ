#!/bin/bash
# check_sweep_status.sh - Enhanced status checker with GPU monitoring

SESSION_PREFIX="ttq-sweep"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

clear

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}TTQ-YOLO Sweep Status${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""

# Check tmux sessions
echo -e "${BLUE}Active tmux sessions:${NC}"
SESSIONS=$(tmux list-sessions 2>/dev/null | grep "${SESSION_PREFIX}")
if [ ! -z "$SESSIONS" ]; then
    echo "$SESSIONS" | while read line; do
        SESSION_NAME=$(echo $line | cut -d: -f1)
        echo "  ✓ $SESSION_NAME"
    done
else
    echo "  No active sweep sessions"
fi

echo ""

# GPU Status
echo -e "${BLUE}GPU Status:${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name gpu_util mem_util mem_used mem_total temp; do
            printf "  GPU %s: %s\n" "$idx" "$name"
            printf "    Utilization: %3d%% GPU, %3d%% Memory\n" "$gpu_util" "$mem_util"
            printf "    Memory: %d/%d MB\n" "$mem_used" "$mem_total"
            printf "    Temperature: %d°C\n" "$temp"
            echo ""
        done
else
    echo "  nvidia-smi not available"
fi

# Process info
echo -e "${BLUE}Python processes:${NC}"
PYTHON_PROCS=$(ps aux | grep "python.*sweep_ttq.py" | grep -v grep)
if [ ! -z "$PYTHON_PROCS" ]; then
    echo "$PYTHON_PROCS" | awk '{printf "  PID %s: GPU %s, CPU %s%%, Mem %s\n", $2, $NF, $3, $4}'
else
    echo "  No sweep processes running"
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${BLUE}Commands:${NC}"
echo "  • Attach to session:  tmux attach -t ${SESSION_PREFIX}-gpu0"
echo "  • Detach:             Ctrl+B, then D"
echo "  • View WandB:         https://wandb.ai/YOUR_USERNAME/ttq-yolo-sweep"
echo "  • Kill all:           ./kill_all_sweeps.sh"
echo "  • Refresh status:     watch -n 5 ./check_sweep_status.sh"
echo -e "${GREEN}=========================================${NC}"
