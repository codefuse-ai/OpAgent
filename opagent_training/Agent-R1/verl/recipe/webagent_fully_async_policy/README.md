# Recipe: WebAgent Fully Async Policy

本目录在上游 [`recipe/fully_async_policy`](../fully_async_policy) 的 Trainer/Rollouter 完全解耦框架之上，扩展出**面向浏览器 Web Agent 场景**的 RL 训练实现，并进一步支持 **Planner + Grounder 双模型** 协同推理/训练。

- 上游 `fully_async_policy` 的设计、参数、训练模式、实验结果，参考同目录 [`README_zh.md`](./README_zh.md)。
- 本文档聚焦本目录的**代码结构**与**模块关系**，帮助快速定位改动落点。

---

## 1. 整体架构

WebAgent 版本在上游异步框架上叠加了三层能力：

1. **Web Agent Loop**：把“LLM ↔ 浏览器”的多轮轨迹适配到 `AgentLoop` 机制，支持 partial rollout 中断与恢复。
2. **Browser Env**：基于 Playwright 的异步浏览器环境、动作空间、自动登录、轨迹处理器。
3. **Reward / Evaluation**：规则匹配 + LLM WebJudge 的异步评估器。

在此之上，**Dual-Model** 分支把单个 policy 拆成两个共享训练流水的模型（Planner 负责 think，Grounder 负责 tool_call），它们拥有独立的 vLLM 推理引擎、独立的 FSDP 训练组与独立的 NCCL 参数同步通道，但共用同一条 RL 轨迹。

### 1.1 单模型链路 (single-model)

```
              ┌──────────────────────── WebAgentFullyAsyncTaskRunner ────────────────────────┐
              │                                                                              │
              │   WebAgentFullyAsyncRollouter                     WebAgentFullyAsyncTrainer  │
              │   ┌──────────────────────────┐                    ┌──────────────────────┐   │
 WebAgentRL   │   │ AgentLoopManager         │                    │ Actor FSDP WG        │   │
   Dataset ──►│   │  └─ AsyncWebAgentLoop ◄──┼──── BrowserEnv ──► │ Critic / Ref / RM    │   │
 (parquet)    │   │       ├─ PENDING         │   (Playwright)     │                      │   │
              │   │       ├─ GENERATING  ◄───┼── vLLM Rollout WG  │  ┌──────────────────┐│   │
              │   │       └─ PROC_TOOLS      │                    │  │ ParameterSync    ││   │
              │   └────────────┬─────────────┘                    │  │ (NCCL)           ││   │
              │                │ RolloutSample                    │  └────────┬─────────┘│   │
              │                ▼                                  │           │          │   │
              │          MessageQueue ────────────────────────────►           │          │   │
              │                                                   └───────────┼──────────┘   │
              └───────────────────────────────────────────────────────────────┼──────────────┘
                                                                              │
                                                         weights update ◄─────┘
```

入口：[`fully_async_main.py`](./fully_async_main.py)

### 1.2 双模型链路 (dual-model)

```
              ┌──────────── DualModelWebAgentFullyAsyncTaskRunner ─────────────────────────────┐
              │                                                                                │
              │   DualModelWebAgentFullyAsyncRollouter                                         │
              │   ┌────────────────────────────────────────────────┐                           │
              │   │ DualModelWebAgentLoopManager                   │                           │
              │   │  └─ DualModelAsyncWebAgentLoop                 │                           │
              │   │       ├─ PLANNING  ──► Planner  vLLM WG  ◄─────┼──┐                        │
              │   │       └─ GROUNDING ──► Grounder vLLM WG  ◄─────┼──┼── DualModelServer      │
              │   │                        (独立 resource pool)    │  │     Manager            │
              │   └────────────────────────┬───────────────────────┘  │                        │
              │                            │                          │                        │
              │                            ▼                          │                        │
              │                     MessageQueue                      │                        │
              │                            │                          │                        │
              │   DualModelWebAgentFullyAsyncTrainer                  │                        │
              │   ┌────────────────────────────────────────────────┐  │                        │
              │   │ split_batch_by_role  ──► Planner  FSDP WG  ◄───┼──┤                        │
              │   │                      └─► Grounder FSDP WG  ◄───┼──┤                        │
              │   └────────────────────────┬───────────────────────┘  │                        │
              │                            │                          │                        │
              │                 DualModelParameterSynchronizer        │                        │
              │                 (planner NCCL group + grounder NCCL group)                     │
              └────────────────────────────┼──────────────────────────┼────────────────────────┘
                                           │                          │
                                           └── weights update ────────┘
```

入口：[`dual_model_main.py`](./dual_model_main.py)（`dual_model.enable=False` 时自动回退到单模型链路）

---

## 2. 代码结构

```
webagent_fully_async_policy/
├── fully_async_main.py                     # 单模型入口 (WebAgentFullyAsyncTaskRunner)
├── fully_async_rollouter.py                # 单模型 Rollouter，继承 FullyAsyncRollouterBase
├── fully_async_trainer.py                  # 单模型 Trainer，继承 FullyAsyncTrainerBase
│
├── dual_model_main.py                      # 双模型入口，可回退到单模型
├── dual_model_rollouter.py                 # 双模型 Rollouter（独立 planner/grounder vLLM WG）
├── dual_model_trainer.py                   # 双模型 Trainer（独立 planner/grounder FSDP WG）
├── dual_model_param_sync.py                # 双模型参数同步器（两套 NCCL collective group）
│
├── _validation_impl.py                     # 共享的 validation 逻辑（断点续验证 / URL 均匀化）
├── agent_rl_dataset.py                     # WebAgentRLDataset + collate_fn（parquet 轨迹数据集）
├── detach_utils.py                         # 轨迹 flatten_steps / batch 组装 / 后处理
│
├── agent_loop/                             # Web Agent Loop 适配层
│   ├── agent_loop.py                       #   FullyAsyncWebAgentLoopManager / Worker
│   ├── web_agent_loop.py                   #   AsyncWebAgentLoop（单模型状态机）
│   ├── dual_model_web_agent_loop.py        #   DualModelAsyncWebAgentLoop（PLANNING / GROUNDING 状态）
│   └── dual_model_server_manager.py        #   DualModelServerManager（按 engine_tag 路由 vLLM 请求）
│
├── browser_env/                            # 浏览器环境（Playwright）
│   ├── async_web_browser_envs.py           #   BrowserActor：Ray Actor 形式的浏览器生命周期管理
│   ├── tool_env.py                         #   WebBrowserEnv：工具执行环境
│   ├── web_browser_tool.py                 #   WebBrowserTool：工具定义 / 动作解析
│   ├── action_space_json.py / actions.py   #   动作空间 Schema 与解析
│   ├── processors.py                       #   观测处理（截图 / DOM / AXTree）
│   ├── auto_login.py / auto_login_asynico.py  # 站点自动登录
│   ├── init_browser_with_restart.py        #   浏览器重启/恢复
│   ├── envs.py / constants.py / env_config.py # 环境基础设施
│   └── javascript/, test_pages/            #   注入脚本与本地测试页
│
├── evaluation_harness/                     # 奖励与评测
│   ├── evaluators.py                       #   基础评估器（精确/模糊匹配 / LLM eval）
│   ├── asyns_evaluators.py                 #   异步并发评估器
│   ├── webjudge/                           #   LLM WebJudge 评估器
│   └── helper_functions.py / image_utils.py / utils.py
│
├── vllm/                                   # vLLM runtime patches
│   ├── gpu_model_runner.py                 #   GPU model runner override
│   └── gpu_model_runner_v011.py            #   vLLM 0.11 版本 patch
│
├── config/                                 # Hydra 配置
│   ├── fully_async_ppo_trainer.yaml        #   FSDP + vLLM 训练配置
│   ├── fully_async_ppo_megatron_trainer.yaml  # Megatron 训练配置
│   └── agent/
│       ├── async_web_agent.yaml            #   单模型 AgentLoop 配置
│       └── dual_model_async_web_agent.yaml #   双模型 AgentLoop 配置
│
├── shell/                                  # 训练启动脚本
│   ├── dapo_7b_async_retool.sh             #   DAPO 7B + 多轮工具
│   ├── dapo_7b_math_fsdp2_*.sh             #   不同资源切分的 FSDP2 DAPO 脚本
│   ├── grpo_30b_a3b_base_math_megatron_*.sh   # 30B MoE + Megatron
│   ├── geo3k_qwen25vl_7b_megatron_4_4.sh   #   VLM 多模态脚本
│   └── runtime_env.yaml                    #   Ray runtime env
│
├── scripts/                                # 辅助脚本
│   ├── vllm_serve.sh                       #   启动 vLLM server
│   ├── model_merge.sh                      #   FSDP 分片合并
│   └── visual_webarena/                    #   VisualWebArena 相关工具
│
├── constant/                               # 常量
└── README_zh.md                            # 上游 fully_async_policy 文档（算法/参数/实验）
```

---

## 3. 核心模块说明

### 3.1 单模型链路

| 文件 | 作用 | 与上游关系 |
| --- | --- | --- |
| `fully_async_main.py` | `WebAgentFullyAsyncTaskRunner`：装配 Rollouter / Trainer / MessageQueue / ParameterSync | 复用上游 `FullyAsyncTaskRunnerBase`、`create_resource_pool_manager` |
| `fully_async_rollouter.py` | 重写 `_get_gen_batch()` / `_validate()` / `_consumer_worker()`，使用 `WebAgentRLDataset` 与 flatten-steps 后处理 | 继承 `FullyAsyncRollouterBase` |
| `fully_async_trainer.py` | 重写 `_process_batch_common()` / `_get_samples_from_queue()`，用 `assemble_batch_from_rollout_samples` 组 batch | 继承 `FullyAsyncTrainerBase` |
| `agent_rl_dataset.py` | parquet 轨迹 → prompt + 轨迹元信息（`config / intent / start_url` 等），过滤超长样本 | 新增 |
| `detach_utils.py` | `postprocess_agent_loop_outputs_flattened_steps` / `merge_rollout_sample_flattened_steps` / `assemble_batch_from_rollout_samples`：把一条多轮轨迹拆成 step 级 `DataProto` 参与 PPO | 新增 |
| `_validation_impl.py` | `webagent_validate()` 等独立函数：**断点续验证**（按 intent+URL 跳过已验过的样本）、URL 均匀化 | 新增，被单/双模型 Rollouter 共用 |

### 3.2 双模型链路

| 文件 | 作用 |
| --- | --- |
| `dual_model_main.py` | `DualModelWebAgentFullyAsyncTaskRunner`：`dual_model.enable=True` 时走双模型路径；否则回退到单模型 |
| `dual_model_rollouter.py` | 由于父类是 `@ray.remote`，无法继承，改为**复制 + 扩展**：额外创建 grounder vLLM worker group，并切换到 `DualModelFullyAsyncWebAgentLoopManager` |
| `dual_model_trainer.py` | 重写 `_create_actor_rollout_classes` / `_create_reference_policy_class` / `_init_models` / `_process_batch_common`：在**完整 batch 上计算 GRPO advantage**，再按 role 字段拆成 planner / grounder 两份独立 old/ref/update |
| `dual_model_param_sync.py` | 维护两套 NCCL collective group（`planner_actor_rollout` / `grounder_actor_rollout`），单次 `sync_weights` 同时同步两个 vLLM 引擎 |

### 3.3 Agent Loop（`agent_loop/`）

`AsyncWebAgentLoop` 的状态机本质是对上游 `AgentLoop` 的扩展，让一条轨迹能被 **partial rollout** 中断与恢复：

```
        ┌──────────┐         ┌─────────────┐         ┌──────────────────┐
  ──►   │ PENDING  │ ──────► │ GENERATING  │ ──────► │ PROCESSING_TOOLS │ ──┐
        └──────────┘         └─────────────┘         └──────────────────┘   │
              ▲                    │                                        │
              │                    ▼ (interrupted)                          │
              │              ┌─────────────┐                                │
              │              │  SUSPENDED  │                                │
              │              └─────────────┘                                │
              └────────────────────────────────────────────────────────────┘
                               (next step or resumed after param sync)
```

**双模型版本**把 `GENERATING` 进一步切成两段：

| 状态 | 推理模型 | 典型 stop tokens |
| --- | --- | --- |
| `PLANNING` | Planner vLLM | `</think>` |
| `GROUNDING` | Grounder vLLM | `</tool_call>`、`</answer>` |

每个 step 会带上 `role ∈ {planner, grounder}`，供 Trainer 在拆 batch 时使用。`DualModelServerManager` 按 `engine_tag` 参数把请求分别路由到 planner / grounder 的服务句柄，并维护独立的 LRU sticky session 与负载均衡。

### 3.4 Browser / Evaluation / vLLM

- **`browser_env/`**：对 Playwright 的异步封装。`BrowserActor` 是 Ray Actor，单实例管理一个浏览器生命周期（含崩溃重启）；`WebBrowserTool` 把 LLM 动作（click/type/scroll/...）解析并下发；`processors.py` 负责观测采集（截图、DOM、可访问性树）。详见 `browser_env/README.md`。
- **`evaluation_harness/`**：奖励计算入口。`evaluators.py` 给出规则类评估器；`asyns_evaluators.py` 将多条轨迹的评估并发化；`webjudge/` 用 LLM 做轨迹级打分。
- **`vllm/`**：两份针对不同 vLLM 版本（0.10 / 0.11）的 `GPUModelRunner` patch，用于在推理路径上注入自定义 hook（sleep/resume、custom sampling 等）。

---

## 4. 两种运行模式

| 模式 | 入口 | 关键开关 | 资源池 |
| --- | --- | --- | --- |
| 单模型 | `fully_async_main.py` | `async_training.*` | Trainer pool + Rollout pool |
| 双模型 | `dual_model_main.py` | `dual_model.enable=True` | Trainer pool + **Planner** Rollout pool + **Grounder** Rollout pool |

- 回退：`dual_model_main.py` 在 `dual_model.enable=False` 时自动走与 `fully_async_main.py` 相同的单模型路径，便于同一份配置在两种模式间切换。
- 配置：参见 [`config/fully_async_ppo_trainer.yaml`](./config/fully_async_ppo_trainer.yaml)（FSDP + vLLM）和 [`config/fully_async_ppo_megatron_trainer.yaml`](./config/fully_async_ppo_megatron_trainer.yaml)（Megatron），AgentLoop 配置放在 [`config/agent/`](./config/agent/)。
- 启动脚本：见 [`shell/`](./shell)；异步训练相关参数（`staleness_threshold`、`trigger_parameter_sync_step`、`partial_rollout` 等）的含义与调参建议见 [`README_zh.md`](./README_zh.md)。

---

## 5. 关键跨 recipe 引用

本目录并非自包含，以下模块**直接复用**上游 `recipe/fully_async_policy`：

- `fully_async_main.FullyAsyncTaskRunnerBase` / `create_resource_pool_manager` / `create_role_worker_mapping`
- `fully_async_rollouter.FullyAsyncRollouterBase`
- `fully_async_trainer.FullyAsyncTrainerBase`
- `message_queue.MessageQueue` / `MessageQueueClient`
- `agent_loop.agent_loop`（base AgentLoop manager / worker）

因此阅读本目录代码时，建议**并排打开上游 recipe**。
