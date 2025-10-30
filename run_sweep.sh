#!/bin/bash
# run_sweep.sh - Launch WandB sweep on single or multiple GPUs using tmux
# Includes virtual environment activation

set -e  # Exit on error

# Default configuration
SESSION_PREFIX="ttq-sweep"
SWEEP_COUNT=20
PROJECT_NAME="ttq-yolo-sweep"
GPUS="0"  # Comma-separated list, e.g., "0,1,2,3"
ENTITY=""

# Virtual environment path (relative to project root)
VENV_PATH="yolo_env"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Help message
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Launch TTQ-YOLO hyperparameter sweep using tmux sessions.

OPTIONS:
    --gpus GPU_LIST         Comma-separated GPU IDs (default: "0")
                           Examples: "0" or "0,1,2,3"

    --count N              Total number of runs across all GPUs (default: 20)

    --sweep-id ID          Join existing sweep (optional)

    --project NAME         WandB project name (default: "ttq-yolo-sweep")

    --entity NAME          WandB entity/username (optional)

    --session PREFIX       tmux session name prefix (default: "ttq-sweep")

    --venv PATH            Path to virtual environment (default: "yolo_env")

    -h, --help            Show this help message

EXAMPLES:
    # Single GPU sweep (20 runs on GPU 0)
    $0 --gpus 0 --count 20

    # Multi-GPU sweep (20 runs distributed across 4 GPUs)
    $0 --gpus 0,1,2,3 --count 20

    # Join existing sweep on multiple GPUs
    $0 --sweep-id abc123 --gpus 0,1 --count 10

    # Custom virtual environment path
    $0 --gpus 0 --count 20 --venv /path/to/venv

TMUX COMMANDS:
    List sessions:    tmux list-sessions
    Attach to GPU 0:  tmux attach -t ttq-sweep-gpu0
    Detach:           Ctrl+B, then D
    Kill session:     tmux kill-session -t ttq-sweep-gpu0
    Kill all:         bash kill_all_sweeps.sh

EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --count)
            SWEEP_COUNT="$2"
            shift 2
            ;;
        --sweep-id)
            SWEEP_ID="$2"
            shift 2
            ;;
        --project)
            PROJECT_NAME="$2"
            shift 2
            ;;
        --entity)
            ENTITY="$2"
            shift 2
            ;;
        --session)
            SESSION_PREFIX="$2"
            shift 2
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Get project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check if virtual environment exists
if [ ! -d "$PROJECT_ROOT/$VENV_PATH" ]; then
    echo -e "${RED}❌ Virtual environment not found: $PROJECT_ROOT/$VENV_PATH${NC}"
    echo "Please create it first or specify correct path with --venv"
    exit 1
fi

# Convert GPU list to array
IFS=',' read -ra GPU_ARRAY <<< "$GPUS"
N_GPUS=${#GPU_ARRAY[@]}

echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}TTQ-YOLO Sweep - Multi-GPU Launcher${NC}"
echo -e "${GREEN}=========================================${NC}"
echo "GPUs: ${GPUS} (${N_GPUS} total)"
echo "Total runs: ${SWEEP_COUNT}"
echo "Session prefix: ${SESSION_PREFIX}"
echo "Project: ${PROJECT_NAME}"
echo "Virtual env: ${VENV_PATH}"
if [ ! -z "$ENTITY" ]; then
    echo "Entity: ${ENTITY}"
fi
if [ ! -z "$SWEEP_ID" ]; then
    echo "Sweep ID: ${SWEEP_ID}"
fi
echo -e "${GREEN}=========================================${NC}"
echo ""

# Calculate runs per GPU
RUNS_PER_GPU=$((SWEEP_COUNT / N_GPUS))
EXTRA_RUNS=$((SWEEP_COUNT % N_GPUS))

echo "Distributing ${SWEEP_COUNT} runs across ${N_GPUS} GPU(s):"
for ((i=0; i<N_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[$i]}
    RUNS=$RUNS_PER_GPU
    if [ $i -eq 0 ]; then
        RUNS=$((RUNS + EXTRA_RUNS))
    fi
    echo "  GPU ${GPU_ID}: ${RUNS} runs"
done
echo ""

# Function to launch sweep on one GPU
launch_on_gpu() {
    local GPU_ID=$1
    local RUNS=$2

    SESSION_NAME="${SESSION_PREFIX}-gpu${GPU_ID}"

    # Check if session exists
    if tmux has-session -t $SESSION_NAME 2>/dev/null; then
        echo -e "${YELLOW}⚠️  Session '$SESSION_NAME' already exists!${NC}"
        echo "   Kill it with: tmux kill-session -t $SESSION_NAME"
        return 1
    fi

    # Create tmux session
    tmux new-session -d -s $SESSION_NAME

    # Set up environment
    tmux send-keys -t $SESSION_NAME "cd $PROJECT_ROOT" C-m

    # Deactivate conda if active
    tmux send-keys -t $SESSION_NAME "conda deactivate 2>/dev/null || true" C-m
    sleep 0.5

    # Activate virtual environment
    tmux send-keys -t $SESSION_NAME "source $VENV_PATH/bin/activate" C-m
    sleep 0.5

    # Set GPU
    tmux send-keys -t $SESSION_NAME "export CUDA_VISIBLE_DEVICES=$GPU_ID" C-m
    tmux send-keys -t $SESSION_NAME "echo -e '${GREEN}✓ Environment activated (GPU $GPU_ID)${NC}'" C-m
    tmux send-keys -t $SESSION_NAME "echo ''" C-m

    # Build sweep command
    SWEEP_CMD="python sweep_ttq.py --count $RUNS --project $PROJECT_NAME"

    if [ ! -z "$ENTITY" ]; then
        SWEEP_CMD="$SWEEP_CMD --entity $ENTITY"
    fi

    if [ ! -z "$SWEEP_ID" ]; then
        SWEEP_CMD="$SWEEP_CMD --sweep-id $SWEEP_ID"
    fi

    # Log info
    tmux send-keys -t $SESSION_NAME "echo -e '${GREEN}Starting $RUNS sweep runs on GPU $GPU_ID${NC}'" C-m
    tmux send-keys -t $SESSION_NAME "echo 'Command: $SWEEP_CMD'" C-m
    tmux send-keys -t $SESSION_NAME "echo ''" C-m

    # Run the sweep
    tmux send-keys -t $SESSION_NAME "$SWEEP_CMD" C-m

    echo -e "${GREEN}✓${NC} Launched session: $SESSION_NAME"
}

# Launch sweeps on all GPUs
echo "Launching sweep agents..."
echo ""

for ((i=0; i<N_GPUS; i++)); do
    GPU_ID=${GPU_ARRAY[$i]}
    RUNS=$RUNS_PER_GPU
    if [ $i -eq 0 ]; then
        RUNS=$((RUNS + EXTRA_RUNS))
    fi

    launch_on_gpu $GPU_ID $RUNS

    # Small delay between launches
    if [ $i -lt $((N_GPUS - 1)) ]; then
        sleep 2
    fi
done

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}✓ All sweep agents launched!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Active tmux sessions:"
tmux list-sessions 2>/dev/null | grep "${SESSION_PREFIX}" || echo "  None"
echo ""
echo "Commands:"
echo "  • Attach to GPU 0:  tmux attach -t ${SESSION_PREFIX}-gpu${GPU_ARRAY[0]}"
echo "  • List sessions:    tmux list-sessions"
echo "  • Detach:           Ctrl+B, then D"
echo "  • Kill all:         bash kill_all_sweeps.sh"
echo ""

if [ ! -z "$SWEEP_ID" ]; then
    SWEEP_URL="https://wandb.ai/${ENTITY:-YOUR_USERNAME}/${PROJECT_NAME}/sweeps/${SWEEP_ID}"
else
    SWEEP_URL="https://wandb.ai/${ENTITY:-YOUR_USERNAME}/${PROJECT_NAME}"
fi

echo "Monitor progress:"
echo "  • WandB: ${SWEEP_URL}"
echo "  • Status: bash check_sweep_status.sh"
echo ""
echo -e "${GREEN}=========================================${NC}"
