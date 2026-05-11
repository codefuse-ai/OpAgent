#!/bin/bash
# =============================================================================
# OpAgent Local WebAgent Evaluation Script
#
# Runs the Reflector-Planner-Grounder-Summary multi-agent system on WebArena tasks.
#
# Usage:
#   # Basic usage (with existing ECS instances)
#   SKIP_ECS_START=1 bash eval/run_local_agent_eval.sh
#
#   # Specify ECS count
#   NUM_ECS=3 SKIP_ECS_START=1 bash eval/run_local_agent_eval.sh
#
#   # Debug mode (skip SSH refresh)
#   VLM_EXP_DEBUG=1 SKIP_ECS_START=1 bash eval/run_local_agent_eval.sh
#
#   # With custom model endpoints
#   REASONING_MODEL=qwen2.5-vl-72b \
#   REASONING_BASE_URL=http://localhost:8000/v1 \
#   GROUNDER_MODEL=qwen2.5-vl-72b \
#   GROUNDER_BASE_URL=http://localhost:8001/v1 \
#   bash eval/run_local_agent_eval.sh
# =============================================================================

set -e

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Agent-R1 root (where this eval/ lives) — working directory for the script
AGENT_R1_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# OpAgent root — needed for PYTHONPATH so that `from opagent.xxx` works
OPAGENT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

cd "$AGENT_R1_ROOT"

# =============================================================================
# Environment Variables
# =============================================================================

# --- Task Configuration ---
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-"opagent_webarena_eval"}"
export SAVE_MODEL_PATH="${SAVE_MODEL_PATH:-"./log/${EXPERIMENT_NAME}/save_model_path"}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-"./log/${EXPERIMENT_NAME}/tensorboard"}"

export TASK_ID="${TASK_ID:-"opagent_webarena_eval"}"
export CONFIG_DIR="${CONFIG_DIR:-"./config_files"}"
export OUTPUT_DIR="${OUTPUT_DIR:-"./log/${EXPERIMENT_NAME}"}"

# --- ECS Configuration ---
export SKIP_ECS_START="${SKIP_ECS_START:-0}"
export ECS_CSV="${ECS_CSV:-"./ecs_instances.csv"}"
export NUM_ECS="${NUM_ECS:-5}"
export ECS_SSH_USERNAME="${ECS_SSH_USERNAME:-"root"}"
export ECS_SSH_PASSWORD="${ECS_SSH_PASSWORD:-""}"

# --- Auth Configuration ---
export WEBARENA_AUTH_PATH="${WEBARENA_AUTH_PATH:-"./log"}"

# --- Browser Configuration ---
export HEADLESS="${HEADLESS:-1}"
export BROWSER_OUTPUT_PATH="${BROWSER_OUTPUT_PATH:-"./log/${EXPERIMENT_NAME}/browser_config"}"
export NUM_BROWSERS="${NUM_BROWSERS:-8}"

# --- Python Path ---
# OpAgent root for `from opagent.xxx` imports
# Agent-R1/eval for `from prompts import ...` (also set in local_agent_eval.py)
export PYTHONPATH="${OPAGENT_ROOT}:${AGENT_R1_ROOT}/eval:${PYTHONPATH:-}"
export VLM_EXP_DEBUG="${VLM_EXP_DEBUG:-1}"
export VERL_LOGGING_LEVEL="${VERL_LOGGING_LEVEL:-INFO}"
export PW_TEST_SCREENSHOT_NO_FONTS_READY="${PW_TEST_SCREENSHOT_NO_FONTS_READY:-1}"

# --- WebArena Environment Variables ---
export WEBHOSTNAME="${WEBHOSTNAME:-"http://localhost"}"
export SHOPPING="${WEBHOSTNAME}:7770"
export SHOPPING_ADMIN="${WEBHOSTNAME}:7780/admin"
export REDDIT="${WEBHOSTNAME}:9999"
export GITLAB="${WEBHOSTNAME}:8023"
export MAP="${WEBHOSTNAME}:3000"
export WIKIPEDIA="${WEBHOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="${WEBHOSTNAME}:4399"
export DATASET="webarena"

# --- Model Configuration ---
# Reasoning model (Reflector/Planner/Summary)
export REASONING_MODEL="${REASONING_MODEL:-"qwen2.5-vl-72b"}"
export REASONING_BASE_URL="${REASONING_BASE_URL:-"http://localhost:8000/v1"}"
export REASONING_API_KEY="${REASONING_API_KEY:-"EMPTY"}"

# Grounder model (coordinate grounding)
export GROUNDER_MODEL="${GROUNDER_MODEL:-"qwen2.5-vl-72b"}"
export GROUNDER_BASE_URL="${GROUNDER_BASE_URL:-"http://localhost:8000/v1"}"
export GROUNDER_API_KEY="${GROUNDER_API_KEY:-"EMPTY"}"

# --- Eval Configuration ---
# Set to 0 to disable WebJudge fallback
export REWARD_COEFF="${REWARD_COEFF:-0}"

# =============================================================================
# Display Configuration
# =============================================================================

echo "=============================================="
echo "OpAgent Local WebAgent Evaluation"
echo "=============================================="
echo "Working dir:  $AGENT_R1_ROOT"
echo "OpAgent root: $OPAGENT_ROOT"
echo "CONFIG_DIR:   $CONFIG_DIR"
echo "OUTPUT_DIR:   $OUTPUT_DIR"
echo "ECS_CSV:      $ECS_CSV"
echo "NUM_ECS:      $NUM_ECS"
echo "WEBHOSTNAME:  $WEBHOSTNAME"
echo ""
echo "Model Configuration:"
echo "  Reasoning: $REASONING_MODEL @ $REASONING_BASE_URL"
echo "  Grounder:  $GROUNDER_MODEL @ $GROUNDER_BASE_URL"
echo "=============================================="

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BROWSER_OUTPUT_PATH"
mkdir -p "$WEBARENA_AUTH_PATH"

# =============================================================================
# Step 1: Initialize Browsers
# =============================================================================

echo ""
echo "[Step 1] Initializing browsers..."

if [ -f "$OPAGENT_ROOT/opagent/init_browser.sh" ]; then
    bash "$OPAGENT_ROOT/opagent/init_browser.sh"
elif [ -f "$OPAGENT_ROOT/opagent/init_browser.py" ]; then
    python "$OPAGENT_ROOT/opagent/init_browser.py"
else
    echo "Warning: Browser init script not found. Assuming browsers are already running."
    echo "If you need to initialize browsers, run: python -m opagent.init_browser"
fi

# =============================================================================
# Step 2: Run Evaluation
# =============================================================================

echo ""
echo "[Step 2] Running OpAgent evaluation..."

# Build command arguments
CMD_ARGS="--dataset-path $CONFIG_DIR --output-dir $OUTPUT_DIR"
CMD_ARGS="$CMD_ARGS --reasoning-model $REASONING_MODEL"
CMD_ARGS="$CMD_ARGS --reasoning-base-url $REASONING_BASE_URL"
CMD_ARGS="$CMD_ARGS --reasoning-api-key $REASONING_API_KEY"
CMD_ARGS="$CMD_ARGS --grounder-model $GROUNDER_MODEL"
CMD_ARGS="$CMD_ARGS --grounder-base-url $GROUNDER_BASE_URL"
CMD_ARGS="$CMD_ARGS --grounder-api-key $GROUNDER_API_KEY"
CMD_ARGS="$CMD_ARGS --webhostname $WEBHOSTNAME"

# ECS CSV file
if [ -f "$ECS_CSV" ]; then
    CMD_ARGS="$CMD_ARGS --ecs-csv $ECS_CSV"
fi

# WebArena auth path
if [ -n "$WEBARENA_AUTH_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --webarena-auth-path $WEBARENA_AUTH_PATH"
fi

# Headless mode
if [ "$HEADLESS" == "1" ]; then
    CMD_ARGS="$CMD_ARGS --headless"
fi

# ECS count
if [ -n "$NUM_ECS" ]; then
    CMD_ARGS="$CMD_ARGS --num-ecs $NUM_ECS"
fi

# Reset web environment
if [ "${RESET_WEB:-0}" == "1" ]; then
    CMD_ARGS="$CMD_ARGS --reset-web"
fi

echo "Running: python eval/local_agent_eval.py $CMD_ARGS"
python eval/local_agent_eval.py $CMD_ARGS

# =============================================================================
# Step 3: Calculate Accuracy
# =============================================================================

echo ""
echo "[Step 3] Calculating accuracy..."

if [ -f "$OPAGENT_ROOT/opagent/calculate_accuracy.py" ]; then
    python "$OPAGENT_ROOT/opagent/calculate_accuracy.py" --output-dir "$OUTPUT_DIR"
elif [ -f "$OPAGENT_ROOT/opagent_training/tools/cal_acc.py" ]; then
    python "$OPAGENT_ROOT/opagent_training/tools/cal_acc.py" "$OUTPUT_DIR"
else
    echo "Accuracy calculator not found. Manual calculation:"
    echo "  python -c \"import json, glob; files = glob.glob('$OUTPUT_DIR/val_*/trajectory.json'); scores = [json.load(open(f))[-1].get('score', 0) for f in files]; print(f'Accuracy: {sum(scores)/len(scores):.4f} ({sum(scores)}/{len(scores)})')\""
fi

echo ""
echo "=============================================="
echo "Evaluation Complete!"
echo "Results: $OUTPUT_DIR"
echo "=============================================="