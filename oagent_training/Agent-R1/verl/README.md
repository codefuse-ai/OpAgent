# Agent-R1: Web Agent 在线强化学习训练系统

基于 [verl](https://github.com/volcengine/verl) 框架的 Web Agent 在线 RL 训练系统，支持 **Planner/Grounder 双模型架构**与**全异步训练**。Agent 通过与真实网页环境（WebArena）的交互，学习自主完成复杂的网页导航任务。

## 核心特性

- **Dual-Model 架构** — Planner 负责推理（`<think>`），Grounder 负责生成动作（`<tool_call>`），独立训练、独立推理
- **全异步训练** — Rollout 和 Training 完全解耦，通过 MessageQueue 流式传输样本，最大化 GPU 利用率
- **WebArena 浏览器环境** — 基于 Playwright 的异步浏览器控制，支持多实例并行交互
- **GRPO 算法** — Group Relative Policy Optimization，适用于 multi-turn agent 场景
- **流式 + 部分 Rollout** — 支持样本级流式传输和参数同步时的在途样本保存

## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        TaskRunner (Ray)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐        ┌──────────────────────────┐   │
│  │      Rollouter       │        │        Trainer           │   │
│  │                      │        │                          │   │
│  │  Planner vLLM ──┐    │  MQ    │  ┌── Planner FSDP       │   │
│  │  Grounder vLLM ──┤   ├───────►│  ├── Grounder FSDP      │   │
│  │  Browser Env ────┘    │        │  └── Reward / Advantage  │   │
│  │                      │        │                          │   │
│  └──────────┬───────────┘        └────────────┬─────────────┘   │
│             │                                  │                 │
│             └──────── ParameterSync (NCCL) ────┘                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 训练流程

1. **Rollouter** 从数据集取 WebArena 任务，启动 Agent Loop（Planner → Grounder → Browser → 截图 → 循环），WebJudge 评估任务完成度并计算 reward，完成的样本推送到 MessageQueue
2. **Trainer** 从 MessageQueue 取样本，计算 GRPO advantage，按角色拆分 batch，独立更新 Planner/Grounder 参数
3. **ParameterSync** 定期通过 NCCL 将更新后的参数同步到 Rollouter 的 vLLM 引擎

### Agent Loop 状态机

每个 rollout sample 执行以下循环，直到任务完成或达到最大轮数：

```
PENDING → PLANNING → GROUNDING → PROCESSING_TOOLS → PLANNING → ... → TERMINATED
             │            │              │
          Planner       Grounder     Playwright
         生成推理       生成动作     执行浏览器操作
        <think>       <tool_call>    截图 → 下一轮观测
```

## 快速开始

### 环境依赖

- Python >= 3.10
- PyTorch >= 2.1
- CUDA >= 12.1
- GPU >= 8 张（推荐 H20 / A100 / H100）

### 安装

```bash
# 1. 安装 verl
pip install -e ".[vllm]"

# 2. 安装浏览器依赖
pip install playwright
playwright install chromium
```

### 单机 8 卡训练

```bash
# 设置必要的环境变量
export BASE_MODEL="/path/to/your/model"          # 如 Qwen2.5-VL-7B-Instruct
export DATASET_PATH="/path/to/webarena/tasks"     # WebArena 任务配置
export VAL_DATASET_PATH="/path/to/val/tasks"
export SAVE_MODEL_PATH="/path/to/save/output"
export WEBHOSTNAME="http://your-webarena-host"    # WebArena 服务地址

# 启动训练
bash recipe/webagent_fully_async_policy/scripts/visual_webarena/run_grpo_verl06_dual_model_async.sh
```

## GPU 布局

以单机 8 卡为例：

| 角色 | GPU 数量 | 用途 |
|------|---------|------|
| Training (FSDP) | 4 | Planner + Grounder 共享训练（colocated workers） |
| Planner Rollout (vLLM) | 2 | Planner 推理引擎 (TP=2) |
| Grounder Rollout (vLLM) | 2 | Grounder 推理引擎 (TP=2) |

多节点场景下，每节点按相同比例分配。总 GPU = `NUM_NODES * GPUS_PER_NODE`。

## 核心配置参数

通过环境变量和 Hydra 命令行参数传入。

### 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BASE_MODEL` | Planner 模型路径 | 需设置 |
| `GROUNDER_MODEL` | Grounder 模型路径 | 与 `BASE_MODEL` 相同 |
| `DATASET_PATH` | 训练数据路径 | 需设置 |
| `VAL_DATASET_PATH` | 验证数据路径 | 需设置 |
| `SAVE_MODEL_PATH` | 模型保存路径 | 需设置 |
| `WEBHOSTNAME` | WebArena 服务地址 | 需设置 |
| `GPUS_PER_NODE` | 每节点 GPU 数 | `8` |
| `NUM_NODES` | 节点数 | `1` |
| `BATCH_SIZE` | 验证 batch size | `2` |
| `NUM_BROWSERS` | 并行浏览器实例数 | `8` |
| `WANDB_MODE` | W&B 模式 | `offline` |

### Dual-Model 参数

| 参数 | 说明 |
|------|------|
| `dual_model.enable` | 启用双模型模式（`True`/`False`） |
| `dual_model.grounder_model_path` | Grounder 模型路径（默认与 Planner 相同，训练后分化） |
| `dual_model.planner_stop_tokens` | Planner 停止 token，如 `["</think>"]` |
| `dual_model.grounder_stop_tokens` | Grounder 停止 token，如 `["</tool_call>", "</answer>"]` |
| `grounder_rollout.n_gpus_per_node` | Grounder 推理引擎的 GPU 数量 |
| `grounder_rollout.tensor_model_parallel_size` | Grounder 的 TP 并行度 |

### 异步训练参数

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `async_training.staleness_threshold` | 样本过期比例阈值（0=同步，>0=异步） | `0.1` |
| `async_training.trigger_parameter_sync_step` | 每 N 步训练后同步参数到 Rollout | `2` |
| `async_training.require_batches` | 每次训练步从队列取的 mini-batch 数 | `1` |
| `async_training.partial_rollout` | 是否保存参数同步时的在途样本 | `False` |

### 训练超参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `actor_rollout_ref.actor.optim.lr` | 学习率 | `1e-6` |
| `actor_rollout_ref.actor.clip_ratio_low` | PPO clip 下界 | `0.2` |
| `actor_rollout_ref.actor.clip_ratio_high` | PPO clip 上界 | `0.28` |
| `actor_rollout_ref.rollout.n` | 每个 prompt 的采样数 | `5` |
| `data.max_prompt_length` | 最大 prompt 长度 | `20000` |
| `data.max_response_length` | 最大回复长度 | `2000` |
| `tool.max_turns` | 最大交互轮数 | `25` |

## 全异步训练模式

系统支持四种训练模式，通过不同参数组合实现：

| 模式 | 参数设置 | 特点 |
|------|---------|------|
| On-policy | `staleness=0, sync_step=1` | 严格同步，最稳定 |
| Stream off-policy | `staleness=0, sync_step>1` | 流式同步，减少空闲 |
| Async + stale | `staleness>0, partial=False` | 异步训练，允许旧样本 |
| Async + partial | `staleness>0, partial=True` | 异步+部分 rollout，最高效 |

使用 Qwen2.5-Math-7B 在 128 卡上验证，全异步模式可达 **2.35x-2.67x** 加速，且不显著影响训练效果。

## 浏览器环境

基于 Playwright 的异步浏览器环境：

- 多实例并行（默认 8 个浏览器）
- 页面截图作为 VLM 视觉输入
- 动作空间：click、type、scroll、select_option、go_back、goto 等
- 自动登录 WebArena 各站点（Reddit、Shopping、GitLab 等）
- 异步执行，支持超时和异常恢复

## 评估系统

- **WebJudge** — 基于 VLM 的网页任务评估器，对比 agent 执行结果与预期目标
- **format_score** — 动作格式正确性评分
- **answer_score** — 任务完成度评分

## 项目结构

```
recipe/webagent_fully_async_policy/
├── dual_model_main.py              # 主入口：TaskRunner 编排双模型初始化
├── dual_model_trainer.py           # 双模型训练器（独立更新 Planner/Grounder）
├── dual_model_rollouter.py         # 管理两套 vLLM 引擎的 Rollouter
├── dual_model_param_sync.py        # 双模型 NCCL 参数同步
├── fully_async_rollouter.py        # 全异步 Rollouter 基类
├── agent_rl_dataset.py             # WebAgent 数据集加载
├── agent_loop/
│   ├── dual_model_web_agent_loop.py  # Agent 状态机（PLANNING/GROUNDING 循环）
│   └── web_agent_loop.py             # 单模型 Agent Loop
├── browser_env/
│   ├── async_web_browser_envs.py   # 异步浏览器环境（Playwright）
│   ├── actions.py                  # 动作解析与执行
│   ├── auto_login.py               # WebArena 站点自动登录
│   └── processors.py               # 观测处理（截图/DOM）
├── evaluation_harness/
│   ├── asyns_evaluators.py         # 异步任务评估
│   └── webjudge/                   # VLM-based 网页评分
├── config/agent/
│   └── dual_model_async_web_agent.yaml  # Agent Loop 配置
└── scripts/
    └── visual_webarena/
        └── run_grpo_verl06_dual_model_async.sh  # 训练启动脚本

recipe/fully_async_policy/
├── fully_async_trainer.py          # 全异步训练器基类
├── fully_async_rollouter.py        # 全异步 Rollouter 基类
├── fsdp_workers.py                 # FSDP Worker 定义
├── message_queue.py                # 样本流式传输队列
└── param_sync.py                   # 参数同步器
```

## 致谢

本项目基于 [verl](https://github.com/volcengine/verl)（ByteDance Seed Team）开发，全异步训练架构参考了以下工作：

- [Magistral](https://arxiv.org/abs/2506.10910)
- [AReaL](https://arxiv.org/abs/2505.24298)
- [StreamRL](https://arxiv.org/abs/2504.15930)
- [AsyncFlow](https://arxiv.org/abs/2507.01663)

## 引用

```bibtex
@article{sheng2024hybridflow,
  title   = {HybridFlow: A Flexible and Efficient RLHF Framework},
  author  = {Guangming Sheng and Chi Zhang and Zilingfeng Ye and Xibin Wu and Wang Zhang and Ru Zhang and Yanghua Peng and Haibin Lin and Chuan Wu},
  year    = {2024},
  journal = {arXiv preprint arXiv:2409.19256}
}
```

## License

Apache License 2.0
