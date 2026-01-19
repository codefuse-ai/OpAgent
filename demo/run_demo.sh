#!/bin/bash

set -e

# --- 任务配置 ---
export EXPERIMENT_NAME="local_webarena_eval_name"
export SAVE_MODEL_PATH="./log/${EXPERIMENT_NAME}/save_model_path"
export TENSORBOARD_DIR="./log/${EXPERIMENT_NAME}/tensorboard" 

export TASK_ID="local_webarena_eval"
export CONFIG_DIR="./test_webarena_conflict"
export OUTPUT_DIR="./log/${EXPERIMENT_NAME}"

# --- 实验控制 ---
export RETEST_FAILED=${RETEST_FAILED:-1} # 是否重测失败的任务 (1: 重测, 0: 不重测)

export ECS_CSV=${ECS_CSV:-"./ecs_instances.csv"}
export NUM_ECS=${NUM_ECS:-5}  # 使用的 ECS 数量
export ECS_TASK_NAME=${ECS_TASK_NAME:-"webarena_task_gemini_annotation"}

# --- 认证配置 ---
export WEBARENA_AUTH_PATH=${WEBARENA_AUTH_PATH:-"./log"}

# --- 浏览器配置 ---
export HEADLESS=${HEADLESS:-1}  # 0: 显示浏览器, 1: 无头模式
export BROWSER_OUTPUT_PATH="./log/${EXPERIMENT_NAME}/browser_config"
export NUM_BROWSERS=${NUM_BROWSERS:-8}

# 环境变量 (和 start_browser.sh 保持一致)
# export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/agent_r1/tool/tools:${PYTHONPATH}"
# Ensure 'demo' package is importable
export PYTHONPATH="$(dirname "$PWD"):${PYTHONPATH}"
export VLM_EXP_DEBUG=${VLM_EXP_DEBUG:-0}

export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}
export PW_TEST_SCREENSHOT_NO_FONTS_READY=${PW_TEST_SCREENSHOT_NO_FONTS_READY:-1}

# --- WebArena 环境变量 ---
# TODO: 配置您的 WebArena 主机地址
export WEBHOSTNAME=${WEBHOSTNAME:-"http://localhost"}
export SHOPPING="${WEBHOSTNAME}:7770"
export SHOPPING_ADMIN="${WEBHOSTNAME}:7780/admin"
export REDDIT="${WEBHOSTNAME}:9999"
export GITLAB="${WEBHOSTNAME}:8023"
export MAP="${WEBHOSTNAME}:3000"
export WIKIPEDIA="${WEBHOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export HOMEPAGE="${WEBHOSTNAME}:4399"
export DATASET="webarena"

#REWARD_COEFF 设置为0不用Webjudge兜底评估
export REWARD_COEFF=${REWARD_COEFF:-0}

# --- 代理设置 ---
# TODO: 如需使用代理，请在环境变量中配置 HTTPS_PROXY 和 HTTP_PROXY
# export HTTPS_PROXY=${HTTPS_PROXY:-"your-proxy:8080"}
# export HTTP_PROXY=${HTTP_PROXY:-"your-proxy:8080"}
export no_proxy="localhost,127.0.0.1,${no_proxy}"

# --- Python 路径 ---
# export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/agent_r1/tool/tools:${PYTHONPATH}"

# =============================================================================
# 显示配置信息
# =============================================================================

echo "=============================================="
echo "Local WebAgent Evaluation (本地模型调用)"
echo "=============================================="
echo "CONFIG_DIR: $CONFIG_DIR"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "ECS_CSV: $ECS_CSV"
echo "NUM_ECS: $NUM_ECS"
echo "WEBARENA_AUTH_PATH: $WEBARENA_AUTH_PATH"
echo "HEADLESS: $HEADLESS"
echo "VLM_EXP_DEBUG: $VLM_EXP_DEBUG"
echo "SKIP_SUCCESSFUL: $SKIP_SUCCESSFUL"
echo ""
echo "模型配置 (在 local_agent_eval.py 中配置):"
echo "  Reasoning: antchat.alipay.com (Qwen3-VL-235B)"
echo "  Grounder: codebot.alipay.com (Qwen2.5-VL-72B-SFT)"
echo "=============================================="

# 创建必要的目录
mkdir -p "$OUTPUT_DIR"
mkdir -p "$BROWSER_OUTPUT_PATH"
mkdir -p "$WEBARENA_AUTH_PATH"

echo "运行本地 Agent 评估..."

# 构建命令参数 (与 local_agent_eval.py 的参数对应)
# --dataset-path: 任务配置文件目录
# --output-dir: 输出目录
# --webarena-auth-path: WebArena认证路径
# --ecs-csv: ECS实例CSV文件
# --headless: 使用无头浏览器
# --num-ecs: 使用的ECS数量

CMD_ARGS="--dataset-path $CONFIG_DIR --output-dir $OUTPUT_DIR"

# ECS CSV 文件
if [ -f "$ECS_CSV" ]; then
    CMD_ARGS="$CMD_ARGS --ecs-csv $ECS_CSV"
fi

# WebArena 认证路径
if [ -n "$WEBARENA_AUTH_PATH" ]; then
    CMD_ARGS="$CMD_ARGS --webarena-auth-path $WEBARENA_AUTH_PATH"
fi

# 无头模式
if [ "$HEADLESS" == "1" ]; then
    CMD_ARGS="$CMD_ARGS --headless"
fi

# ECS 数量
if [ -n "$NUM_ECS" ]; then
    CMD_ARGS="$CMD_ARGS --num-ecs $NUM_ECS"
fi

# 是否重测失败的任务
if [ "$RETEST_FAILED" == "1" ]; then
    CMD_ARGS="$CMD_ARGS --retest-failed"
fi
bash ./init_browser.sh
python -u local_agent_eval.py $CMD_ARGS


echo ""
echo "=============================================="
