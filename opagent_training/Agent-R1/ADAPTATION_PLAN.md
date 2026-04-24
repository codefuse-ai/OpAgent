# Agent-R1 适配到 verl_06 fully_async_policy 计划

## 一、概述

### 1.1 目标
将基于旧verl仓库开发的agent_r1代码适配到新verl_06仓库的fully_async_policy全异步策略，实现高效、清晰的代码迁移。

### 1.2 核心差异分析

#### 旧架构（agent_r1）
- **训练模式**: 同步训练，使用`RayAgentTrainer`
- **数据流**: 单进程控制，rollout和train在同一流程中
- **奖励计算**: 需要环境对象（envs），与环境交互紧密耦合
- **环境管理**: 在训练循环中管理WebBrowserEnv/ToolEnv生命周期
- **支持特性**: 多轮对话、工具调用、浏览器环境

#### 新架构（fully_async_policy）
- **训练模式**: 完全异步，`FullyAsyncRollouter`和`FullyAsyncTrainer`分离
- **数据流**: 通过`MessageQueue`传递样本，流式处理
- **奖励计算**: 使用`AbstractRewardManager`接口，只接收`DataProto`
- **环境管理**: 需要在异步架构中重新设计环境管理
- **支持特性**: 流式生成、异步训练、partial rollout

## 二、适配策略

### 2.1 总体原则
1. **最小化修改**: 尽量复用现有代码，只做必要的适配
2. **清晰分层**: 保持代码结构清晰，便于维护
3. **渐进式迁移**: 分阶段实现，确保每个阶段可测试
4. **保持兼容**: 适配过程中保持向后兼容，便于回退

### 2.2 适配架构设计

```
verl_06/recipe/r1_async/
├── fully_async_main.py          # 主入口，基于fully_async_policy/fully_async_main.py
├── fully_async_rollouter.py    # Rollouter适配，集成agent环境
├── fully_async_trainer.py      # Trainer适配，保持原有训练逻辑
├── agent_reward_manager.py     # 奖励管理器适配层
├── agent_env_manager.py        # 环境管理器，管理env生命周期
├── config/
│   └── fully_async_agent_trainer.yaml
└── README.md
```

## 三、详细适配计划

### 阶段1: 奖励管理器适配（核心难点）

#### 3.1.1 问题分析
- **现状**: agent_r1的`RewardManager`和`WebAgentRewardManager`需要envs参数
- **目标**: 适配到`AbstractRewardManager`接口，只接收`DataProto`
- **挑战**: 环境对象需要在奖励计算时可用，但异步架构中样本是流式传递的

#### 3.1.2 解决方案
创建`AgentRewardManager`适配层：

```python
# recipe/r1_async/agent_reward_manager.py
class AgentRewardManager(AbstractRewardManager):
    """
    适配agent_r1的奖励管理器到AbstractRewardManager接口
    通过env_id在样本中关联环境对象
    """
    def __init__(self, tokenizer, num_examine, env_manager, **kwargs):
        # env_manager负责管理环境对象池
        self.env_manager = env_manager
        # 复用原有的奖励计算逻辑
        self.reward_fn = WebAgentRewardManager(tokenizer, num_examine)
    
    def __call__(self, data: DataProto, return_dict=False):
        # 从data中提取env_id，获取对应的env对象
        envs = self.env_manager.get_envs_for_batch(data)
        # 调用原有的奖励计算逻辑
        reward_tensor, answer_lst, format_lst = self.reward_fn(envs, data)
        # 返回符合接口要求的格式
        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "answer_lst": answer_lst,
                "format_lst": format_lst
            }
        return reward_tensor
```

#### 3.1.3 环境管理器设计
```python
# recipe/r1_async/agent_env_manager.py
class AgentEnvManager:
    """
    管理agent环境的生命周期
    在异步架构中，环境对象需要与样本关联
    """
    def __init__(self, config, env_factory):
        self.config = config
        self.env_factory = env_factory  # 创建env的函数
        self.browser_actor_pool = {}  # env_id -> env对象
        self.env_metadata = {}  # env_id -> 元数据（创建时间、使用次数等）
    
    def create_env(self, env_id):
        """创建新环境"""
        env = self.env_factory(self.config)
        self.browser_actor_pool[env_id] = env
        return env
    
    def get_env(self, env_id):
        """获取环境对象"""
        if env_id not in self.browser_actor_pool:
            return self.create_env(env_id)
        return self.browser_actor_pool[env_id]
    
    def get_envs_for_batch(self, data: DataProto):
        """为batch获取对应的env列表"""
        envs = []
        for i in range(len(data)):
            env_id = data[i].non_tensor_batch.get('env_id', f'env_{i}')
            envs.append(self.get_env(env_id))
        return envs
    
    def cleanup_env(self, env_id):
        """清理环境"""
        if env_id in self.browser_actor_pool:
            # 执行清理逻辑
            del self.browser_actor_pool[env_id]
```

### 阶段2: Rollouter适配

#### 3.2.1 主要修改点
1. **环境初始化**: 在rollout开始前创建环境
2. **样本生成**: 在生成样本时关联env_id
3. **奖励计算**: 集成环境管理器，在rollout阶段计算奖励

#### 3.2.2 实现要点
```python
# recipe/r1_async/fully_async_rollouter.py
class FullyAsyncAgentRollouter(FullyAsyncRollouter):
    def __init__(self, ..., env_manager, **kwargs):
        super().__init__(...)
        self.env_manager = env_manager
    
    def _generate_samples(self, prompts):
        # 为每个prompt创建/获取环境
        env_ids = []
        for prompt in prompts:
            env_id = self._get_or_create_env(prompt)
            env_ids.append(env_id)
        
        # 生成样本，在DataProto中保存env_id
        samples = self.rollout_wg.generate_sequences(prompts)
        for i, sample in enumerate(samples):
            sample.non_tensor_batch['env_id'] = env_ids[i]
        
        # 计算奖励（如果需要）
        if self.reward_fn:
            envs = [self.env_manager.get_env(env_id) for env_id in env_ids]
            rewards = self.reward_fn(envs, samples)
        
        return samples
```

### 阶段3: Trainer适配

#### 3.3.1 主要修改点
1. **奖励计算**: 使用适配后的`AgentRewardManager`
2. **环境管理**: 在训练过程中管理环境状态
3. **多轮对话**: 支持多轮对话的状态管理

#### 3.3.2 实现要点
```python
# recipe/r1_async/fully_async_trainer.py
class FullyAsyncAgentTrainer(FullyAsyncTrainer):
    def __init__(self, ..., env_manager, **kwargs):
        super().__init__(...)
        self.env_manager = env_manager
    
    def _train_step(self, samples):
        # 从MessageQueue获取样本
        batch = self._get_samples_from_queue()
        
        # 计算奖励（使用适配后的奖励管理器）
        rewards = self.reward_fn(batch)
        
        # 执行训练
        loss = self._compute_loss(batch, rewards)
        
        # 更新环境状态（如果需要）
        self._update_env_states(batch)
        
        return loss
```

### 阶段4: 主入口适配

#### 3.4.1 实现要点
```python
# recipe/r1_async/fully_async_main.py
@ray.remote(num_cpus=1)
class FullyAsyncAgentTaskRunner(FullyAsyncTaskRunner):
    def _initialize_components(self, config):
        # 初始化环境管理器
        from recipe.r1_async.agent_env_manager import AgentEnvManager
        env_factory = self._create_env_factory(config)
        self.env_manager = AgentEnvManager(config, env_factory)
        
        # 初始化奖励管理器（使用适配层）
        from recipe.r1_async.agent_reward_manager import AgentRewardManager
        reward_fn = AgentRewardManager(
            tokenizer=self.components["tokenizer"],
            num_examine=0,
            env_manager=self.env_manager
        )
        
        # 初始化Rollouter和Trainer
        self._create_rollouter(config, env_manager=self.env_manager)
        self._create_trainer(config, env_manager=self.env_manager)
```

## 四、关键技术点

### 4.1 环境与样本的关联
**方案**: 在`DataProto.non_tensor_batch`中保存`env_id`，通过env_id在需要时获取环境对象。

**优点**:
- 不改变DataProto的核心结构
- 环境对象可以延迟创建和复用
- 支持环境池化管理

### 4.2 多轮对话支持
**方案**: 在`env_metadata`中保存对话历史，每次rollout时恢复对话状态。

**实现**:
```python
class AgentEnvManager:
    def save_conversation_state(self, env_id, conversation_state):
        """保存对话状态"""
        self.env_metadata[env_id]['conversation'] = conversation_state
    
    def restore_conversation_state(self, env_id):
        """恢复对话状态"""
        return self.env_metadata[env_id].get('conversation', [])
```

### 4.3 工具调用支持
**方案**: 工具调用逻辑保持不变，通过环境对象执行。在异步架构中，工具调用在rollout阶段完成。

### 4.4 浏览器环境支持
**方案**: WebBrowserEnv的异步特性（BrowserActor）可以很好地适配到异步架构中。

## 五、实施步骤

### Step 1: 创建基础结构（1-2天）
1. 创建`recipe/r1_async/`目录
2. 复制`fully_async_policy`的基础文件
3. 创建适配层框架

### Step 2: 实现环境管理器（2-3天）
1. 实现`AgentEnvManager`
2. 测试环境创建和获取
3. 实现环境池化

### Step 3: 实现奖励管理器适配（3-4天）
1. 实现`AgentRewardManager`
2. 适配`RewardManager`和`WebAgentRewardManager`
3. 测试奖励计算逻辑

### Step 4: 适配Rollouter（3-4天）
1. 修改`FullyAsyncRollouter`，集成环境管理
2. 实现样本生成时的环境关联
3. 测试rollout流程

### Step 5: 适配Trainer（2-3天）
1. 修改`FullyAsyncTrainer`，使用适配后的奖励管理器
2. 实现训练循环中的环境管理
3. 测试训练流程

### Step 6: 集成测试（3-5天）
1. 端到端测试
2. 性能对比测试
3. 修复bug和优化

### Step 7: 文档和清理（1-2天）
1. 编写使用文档
2. 代码清理和注释
3. 提交PR

**总计**: 约15-23天

## 六、风险评估与应对

### 6.1 风险点
1. **环境状态管理复杂**: 异步架构中环境状态可能不同步
   - **应对**: 使用版本号或时间戳管理环境状态

2. **性能影响**: 环境对象池化可能带来内存开销
   - **应对**: 实现环境对象的生命周期管理，及时清理

3. **多轮对话状态**: 对话历史在异步架构中难以管理
   - **应对**: 将对话历史序列化保存在样本中

4. **工具调用时序**: 工具调用结果需要与样本正确关联
   - **应对**: 使用唯一ID关联工具调用和样本

### 6.2 回退方案
如果适配遇到重大问题，可以：
1. 保持旧代码不变，新代码作为可选方案
2. 分阶段迁移，先迁移简单场景
3. 使用特性开关控制新旧代码路径

## 七、代码组织建议

### 7.1 目录结构
```
Agent-R1/
├── agent_r1/                    # 原有代码保持不变
│   ├── src/
│   └── llm_agent/
├── verl_06/
│   └── recipe/
│       ├── fully_async_policy/  # 原始实现
│       └── r1_async/            # 新的适配实现
│           ├── fully_async_main.py
│           ├── fully_async_rollouter.py
│           ├── fully_async_trainer.py
│           ├── agent_reward_manager.py
│           ├── agent_env_manager.py
│           └── config/
└── ADAPTATION_PLAN.md           # 本文档
```

### 7.2 代码复用策略
1. **直接复用**: `agent_r1/src/reward_score/`中的奖励计算逻辑
2. **适配复用**: `agent_r1/src/main_agent.py`中的环境创建逻辑
3. **参考复用**: `fully_async_policy`中的异步架构

## 八、测试策略

### 8.1 单元测试
- 环境管理器测试
- 奖励管理器适配测试
- 样本生成测试

### 8.2 集成测试
- 端到端训练测试
- 多轮对话测试
- 工具调用测试

### 8.3 性能测试
- 与旧版本性能对比
- 内存使用监控
- 训练速度对比

## 九、后续优化方向

1. **环境复用优化**: 实现更智能的环境对象复用策略
2. **异步工具调用**: 进一步优化工具调用的异步性能
3. **状态压缩**: 优化对话状态的存储和传输
4. **监控和调试**: 添加更完善的监控和调试工具

## 十、总结

本适配计划采用**适配层模式**，在保持原有代码逻辑不变的前提下，通过适配层将agent_r1的特性集成到fully_async_policy架构中。这种方式既保证了代码的清晰性，又最大化了代码复用，降低了适配风险。

关键成功因素：
1. **环境管理器的设计**: 这是整个适配的核心
2. **奖励管理器的适配**: 需要无缝对接原有逻辑
3. **渐进式实施**: 分阶段实施，确保每个阶段可测试
4. **充分的测试**: 确保功能正确性和性能















