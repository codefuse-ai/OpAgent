#!/bin/bash
# ===============================================================
# Dual-Model Planner/Grounder Async Training Launch Script
#
# This script launches training with TWO separate models:
#   - Planner: generates <think>...</think> reasoning blocks
#   - Grounder: generates <tool_call>...</tool_call> action blocks
#
# GPU Layout (example with 16 GPUs per node):
#   Training GPUs (shared FSDP):    8 GPUs  (planner + grounder colocated)
#   Planner rollout (vLLM):         4 GPUs  (TP=4)
#   Grounder rollout (vLLM):        4 GPUs  (TP=4)
#
# Key difference from single-model:
#   - dual_model.enable=True
#   - dual_model.grounder_model_path (optional, defaults to same as planner)
#   - grounder_rollout.n_gpus_per_node for grounder vLLM engine
#   - Agent loop uses "dual_model_async_web_agent" with state machine split
# ===============================================================

ulimit -c 0

export GPU_MODEL_RUNNER_SCRIPTS=${GPU_MODEL_RUNNER_SCRIPTS:-"gpu_model_runner.py"}
# Skip gpu_model_runner.py replacement - the custom file is incompatible with vllm 0.11
# (missing PoolerOutput, Mamba2AttentionBackend, etc.). The NVTX profiling it adds is optional.
echo "[INFO] Using native vllm gpu_model_runner.py (skipping custom replacement for vllm 0.11 compatibility)"
rm -rf ${TMPDIR:-/tmp}/gui_agent_tmp/*
echo "端口:$HOST_PORTS"
export HOST_PORTS=$HOST_PORTS
export VLLM_USE_V1=${VLLM_USE_V1:-1}
export TASK_ID=${TASK_ID:-"test_dual_model_async"}
export VLM_EXP_DEBUG=${VLM_EXP_DEBUG:-0}
export DIST_WEBBROWSER=${DIST_WEBBROWSER:-0}
export VERL_LOGGING_LEVEL=${VERL_LOGGING_LEVEL:-INFO}

# Workaround: Force CUDA detection in containers where driver version check fails
# but GPUs are physically present (e.g., old host driver visible from container)
export VERL_FORCE_CUDA=${VERL_FORCE_CUDA:-1}

# --- 基础配置 ---
export WANDB_MODE=${WANDB_MODE:-offline}
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export PW_TEST_SCREENSHOT_NO_FONTS_READY=${PW_TEST_SCREENSHOT_NO_FONTS_READY:-1}

export OBSERVATION_TYPE=${OBSERVATION_TYPE:-"image"}

# ================= Dual-Model Config =================
# Planner model (also used as default if grounder not specified)
export BASE_MODEL=${BASE_MODEL:-'/path/to/your/model'}
# Grounder model (set to same as BASE_MODEL to start from same weights, diverges during training)
export GROUNDER_MODEL=${GROUNDER_MODEL:-${BASE_MODEL}}

export PROJECT_NAME=${PROJECT_NAME:-'dual_model_async_webagent'}
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"DualModel_Planner_Grounder"}

export WEBARENA_AUTH_PATH=${WEBARENA_AUTH_PATH:-'./config_files/'}

# --- 轨迹保存 ---
export TRAJECTORY_SAVE_FREQ=${TRAJECTORY_SAVE_FREQ:-1}
export TRAJECTORY_SAVE_ENABLED=${TRAJECTORY_SAVE_ENABLED:-true}
export DATASET_PATH=${DATASET_PATH:-'./data/train'}
export VAL_DATASET_PATH=${VAL_DATASET_PATH:-'./data/val'}
export SAVE_MODEL_PATH=${SAVE_MODEL_PATH:-'./output/dual_model_async'}

# --- WebArena & 网络代理配置 ---
export WEBHOSTNAME=${WEBHOSTNAME:-"http://your-webarena-host"}
export SHOPPING=${SHOPPING:-"${WEBHOSTNAME}:7770"}
export SHOPPING_ADMIN=${SHOPPING_ADMIN:-"${WEBHOSTNAME}:7780/admin"}
export REDDIT=${REDDIT:-"${WEBHOSTNAME}:9999"}
export GITLAB=${GITLAB:-"${WEBHOSTNAME}:8023"}
export MAP=${MAP:-"${WEBHOSTNAME}:3000"}
export WIKIPEDIA=${WIKIPEDIA:-"${WEBHOSTNAME}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"}
export HOMEPAGE=${HOMEPAGE:-"${WEBHOSTNAME}:4399"}

export DATASET=${DATASET:-"webarena"}
export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-False}
export WEBARENA_PROXY=${WEBARENA_PROXY:-""}
export HTTPS_PROXY=${HTTPS_PROXY:-""}
export HTTP_PROXY=${HTTP_PROXY:-""}
export no_proxy="localhost,127.0.0.1,${no_proxy}"
echo "NO_PROXY:$NO_PROXY"

export PYTHONPATH="./agent_r1/tool/tools:${PYTHONPATH}"
export BATCH_SIZE=${BATCH_SIZE:-2}
echo "BATCH_SIZE: "${BATCH_SIZE}
export NUM_NODES=${NUM_NODES:-1}
export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export SLOW_MO=${SLOW_MO:-300}
export PLAYWRIGHT_BROWSERS_PATH=${PLAYWRIGHT_BROWSERS_PATH:-"/root/.cache/ms-playwright"}
export MAX_CONCURRENT_WORKERS=${MAX_CONCURRENT_WORKERS:-32}
export DATA_SFUFFLE=${DATA_SFUFFLE:-False}
export TENSORBOARD_DIR=${TENSORBOARD_DIR:-$SAVE_MODEL_PATH/tensorboard/}
echo "NO_PROXY:$NO_PROXY"
mkdir -p $SAVE_MODEL_PATH
mkdir -p $TENSORBOARD_DIR
echo "任务ID: $TASK_ID"
echo "PYTHONPATH: "${PYTHONPATH}
echo "OBSERVATION_TYPE: "${OBSERVATION_TYPE}

export NUM_BROWSERS=${NUM_BROWSERS:-8}
export BROWSER_OUTPUT_PATH=${SAVE_MODEL_PATH}/browser_config/${TASK_ID}
echo "浏览器配置将输出到: ${BROWSER_OUTPUT_PATH}"
rm -rf ${BROWSER_OUTPUT_PATH}/*
bash ./recipe/webagent_fully_async_policy/scripts/env/init_browser.sh
echo "init browser done"


ls

# ================= algorithm =================
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_turns=25
max_prompt_length=20000
max_response_length=2000
actor_lr=1e-6

# ================= GPU Layout for Dual-Model =================
# Total GPUs = training + planner_rollout + grounder_rollout
# IMPORTANT: With 8 GPUs/node, the layout is:
#   - 4 GPUs for training (trainer_pool)
#   - 2 GPUs for planner vLLM (rollout_pool)
#   - 2 GPUs for grounder vLLM (grounder_rollout_pool)
# However, Ray placement groups may schedule these on overlapping physical GPUs!
# This requires gpu_memory_utilization to be set conservatively.

# Training GPUs (shared between planner & grounder FSDP via colocated workers)
n_gpus_training_per_node=$((GPUS_PER_NODE / 2))

# Planner rollout GPUs
n_gpus_planner_rollout_per_node=$((GPUS_PER_NODE / 4))

# Grounder rollout GPUs
n_gpus_grounder_rollout_per_node=$((GPUS_PER_NODE / 4))

# Verify total doesn't exceed available
total_gpus_per_node=$((n_gpus_training_per_node + n_gpus_planner_rollout_per_node + n_gpus_grounder_rollout_per_node))
echo "GPU Layout: ${n_gpus_training_per_node} training + ${n_gpus_planner_rollout_per_node} planner rollout + ${n_gpus_grounder_rollout_per_node} grounder rollout = ${total_gpus_per_node}/${GPUS_PER_NODE}"

if [ $total_gpus_per_node -gt $GPUS_PER_NODE ]; then
    echo "ERROR: Total GPUs ($total_gpus_per_node) exceeds available ($GPUS_PER_NODE)"
    exit 1
fi

# --- Global training config ---
total_training_gpus=$((NUM_NODES * n_gpus_training_per_node))

# --- Inference TP ---
# Planner TP = planner rollout GPUs per node
planner_infer_tp=$n_gpus_planner_rollout_per_node

if [[ $BASE_MODEL == *"7B"* ]]; then
    planner_infer_tp=4
fi
if [ $planner_infer_tp -gt $n_gpus_planner_rollout_per_node ]; then
    planner_infer_tp=$n_gpus_planner_rollout_per_node
fi

# Grounder TP = grounder rollout GPUs per node (can be different if models differ in size)
grounder_infer_tp=$n_gpus_grounder_rollout_per_node

if [[ $GROUNDER_MODEL == *"7B"* ]]; then
    grounder_infer_tp=4
fi
if [ $grounder_infer_tp -gt $n_gpus_grounder_rollout_per_node ]; then
    grounder_infer_tp=$n_gpus_grounder_rollout_per_node
fi

echo "Planner TP: ${planner_infer_tp}, Grounder TP: ${grounder_infer_tp}"

# FSDP config
# Use HYBRID_SHARD with fsdp_size=8 (shard across 2 nodes) for 32B dual-model
# fsdp_size=4 (single node) is too small for colocated 32B×2 models
# fsdp_size=64 (all nodes) has excessive cross-node communication
train_sp=1
fsdp_size=8  # HYBRID_SHARD: shard across 2 nodes (4 GPUs/node × 2 nodes)
offload=True

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 1 ))

# ================= async policy =================
rollout_name="vllm"
rollout_mode="async"

ppo_mini_batch_size=$((BATCH_SIZE / 2))
gen_prompt_bsz=1
n_resp_per_prompt=5
n_resp_per_prompt_val=5
total_rollout_steps=$(((64*250)))
test_freq=200
staleness_threshold=0.1
trigger_parameter_sync_step=2
require_batches=1
partial_rollout=False

python3 -m recipe.webagent_fully_async_policy.dual_model_main \
    algorithm.adv_estimator=$adv_estimator \
    +algorithm.use_process_rewards=True \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files=${DATASET_PATH} \
    data.val_files=${VAL_DATASET_PATH} \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.val_batch_size=${BATCH_SIZE} \
    +data.val_max_concurrent_batches=1 \
    data.return_raw_chat=True \
    data.shuffle=${DATA_SFUFFLE} \
    +data.use_custom_tool_format_func=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=38000 \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    +data.max_response_length_single_turn=500 \
    +data.max_tool_response_length=1500 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key='screenshot' \
    data.prompt_key='prompt' \
    actor_rollout_ref.nccl_timeout=10800 \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.policy_loss.loss_mode=kl_cov \
    actor_rollout_ref.actor.policy_loss.kl_cov_ratio=0.0002 \
    actor_rollout_ref.actor.policy_loss.ppo_kl_coef=1 \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=-1 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=cosine \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=$offload \
    actor_rollout_ref.rollout.name=$rollout_name \
    actor_rollout_ref.rollout.mode=$rollout_mode \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$planner_infer_tp \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_turns \
    actor_rollout_ref.rollout.multi_turn.format=hermes \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.agent.agent_loop_config_path='recipe/webagent_fully_async_policy/config/agent/dual_model_async_web_agent.yaml' \
    trainer.logger=['console','tensorboard'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=${VAL_BEFORE_TRAIN} \
    trainer.log_val_generations=50 \
    trainer.save_freq=20 \
    trainer.test_freq=$test_freq \
    trainer.default_local_dir=$SAVE_MODEL_PATH \
    trainer.total_epochs=10 \
    trainer.critic_warmup=0 \
    trainer.resume_mode=auto \
    trainer.nnodes=$NUM_NODES \
    trainer.n_gpus_per_node=$n_gpus_training_per_node \
    trainer.balance_batch=False \
    rollout.nnodes=$NUM_NODES \
    rollout.n_gpus_per_node=$n_gpus_planner_rollout_per_node \
    rollout.total_rollout_steps=$total_rollout_steps \
    rollout.total_epochs=20 \
    rollout.test_freq=$test_freq \
    +grounder_rollout.n_gpus_per_node=$n_gpus_grounder_rollout_per_node \
    +grounder_rollout.nnodes=$NUM_NODES \
    +grounder_rollout.tensor_model_parallel_size=$grounder_infer_tp \
    +dual_model.enable=True \
    +dual_model.grounder_model_path=$GROUNDER_MODEL \
    +dual_model.planner_stop_tokens='["</think>"]' \
    +dual_model.grounder_stop_tokens='["</tool_call>", "</answer>"]' \
    async_training.staleness_threshold=$staleness_threshold \
    async_training.trigger_parameter_sync_step=$trigger_parameter_sync_step \
    async_training.require_batches=$require_batches \
    async_training.partial_rollout=$partial_rollout \
    async_training.compute_prox_log_prob=True \
    async_training.use_rollout_log_probs=True \
    +async_training.trajectory_save_freq=$TRAJECTORY_SAVE_FREQ \
    +tool.max_turns=$max_turns \
    +tool.env='webbrowser' \
    +tool.webbrowser.render=False \
    +tool.webbrowser.slow_mo=${SLOW_MO} \
    +tool.webbrowser.observation_type=${OBSERVATION_TYPE} \
    +tool.webbrowser.current_viewport_only=False \
    +tool.webbrowser.viewport_width=1280 \
    +tool.webbrowser.viewport_height=720 \
    +tool.webbrowser.save_trace_enabled=True \
    +tool.webbrowser.sleep_after_execution=3.0 \
    +tool.webbrowser.caption_image_fn=None
