# OAgent 

![CodefuseLogo](./assets/github-codefuse-logo-update.jpg)

## Contents
- [News](#news)
- [Introduction](#introduction)
  - [Framework](#1-framework)
  - [Key Modules](#2-key-modules)
  - [Prompt System](#3-prompt-system)
  - [Key Features](#4-key-features)
- [Usage](#usage)
  - [Installation](#installation)
  - [Quick Start](#quick-start)
  - [Command Line Options](#command-line-options)
  - [Interactive Mode](#interactive-mode)
- [Citation](#citation)
  

## News
🔥🔥🔥 [2026/01/22] We are pleased to announce that Oagent achieves a remarkable 71.6% resolve rate on the [Webarena](https://webarena.dev/) leaderboard.
- 🤖 **Huggingface**: [codefuse-ai/OAgent](https://huggingface.co/codefuse-ai/OpAgent-32B)
- 🤖 **Modelscope**: [codefuse-ai/OAgent](https://modelscope.cn/models/codefuse-ai/OpAgent-32B)


## Introduction
This document describes the structure of the demo WebAgent framework implemented in the `./demo/local_agent_eval.py` script. This framework aims to execute and evaluate automated tasks in real Web environments (such as the WebArena Shopping environment) via local/remote model calls.

### 1. Framework 

This Agent adopts a modular **Planner-Grounder-Reflector-Summary** architecture. The entire system consists of a task scheduler, multi-threaded Workers, browser environment management, and core Agent logic.

#### Agent Loop

The execution flow of the Agent is a closed-loop system, mainly containing the following steps:

1.  **Observation**: Acquire the current webpage screenshot.
2.  **Reflector(Gemini3-Pro)**:
    *   Analyzes the execution result of the previous action.
    *   Checks if the task is completed (`is_task_done`).
    *   Collects key information (Notes) to satisfy user requests.
    *   Provides feedback signals to the Planner.
3.  **Planner(Gemini3-Pro)**:
    *   Receives feedback from the Reflector, the current screenshot, and domain expert tips (`tips`).
    *   Generates the next high-level instruction (`instruction`) and action type (`action_type`).
    *   **Expert Strategy**: Dynamically injects expert knowledge and navigation strategies for specific domains (e.g., Adobe Commerce Admin).
4.  **Grounder(PostTraining-Qwen2.5-VL-72B)**:
    *   We collected millions of data points and trained a version of Grounder based on Qwen2.5-VL-72B through post-training (SFT and RL).
    *   Receives instructions and the current screenshot from the Planner.
    *   Uses a Vision Language Model (VLM) to output specific page coordinates (`coords`) or operation parameters.
5.  **Action Execution(Playwright)**:
    *   Executes specific operations (Click, Type, Scroll, Select Option, etc.) via Playwright.
6.  **Summary(Gemini3-Pro)**:
    *   Generates the final answer based on execution history and collected information at the end of the task.

---

### 2. Key Modules

#### 2.1 `LocalWebAgent` Class
The main body of the Agent, responsible for maintaining task status, calling various model modules, and executing the main loop.

*   **State Maintenance**: `steps` (history steps), `marked_notes` (collected info), `last_screenshot`.
*   **Model Calls**:
    *   `call_reflector`: Calls the reasoning model to judge status.
    *   `call_planner`: Calls the reasoning model to generate plans.
    *   `call_grounder`: Calls the visual model (usually an SFT model) to get precise coordinates.
    *   `call_summary`: Generates the final answer.
*   **Strategy Injection**: `get_domain_specific_tips` dynamically loads operation guides for different sites like Shopping/Admin/Map based on the current URL.

#### 2.2 `LocalModelCaller` Class
A unified model call interface encapsulating requests to different backend services:
*   **MatrixLLM / Gemini**: Used for reasoning (Planner/Reflector).
*   **CodeBot / OpenAI SDK**: Used for Grounder (Qwen-VL, etc.).
*   **HTTP**: General HTTP calls.
*   **AFTS Tool Integration**: Automatically handles image uploads, converting Base64 to URLs for specific models.

#### 2.3 `BrowserActor` & Distributed Execution
*   **BrowserActor**: Encapsulates the Playwright Browser instance, supporting browser connection management across threads/processes.
*   **Worker**: Multi-threaded workflow, where each Worker binds an ECS instance IP and a Browser Endpoint.
*   **Environment Refresh**: Automatically handles SSH connections, Cookie injection, and ECS website status reset before tasks start.

---


### 3. Prompt System

The framework defines four core Prompt templates guiding different Agent roles:

*   **`REFLECTION_PROMPT`**: Emphasizes "based on observed facts", responsible for verifying task success criteria, detecting infinite loops, and collecting structured data.
*   **`PLANNER_PROMPT`**: Responsible for generating atomic operation instructions. Includes detailed action definitions (scroll, click, type, etc.) and core principles (priority search, table pagination checks, etc.).
*   **`GROUNDER_PROMPT`**: Concise visual instructions requiring the model to output `<tool_call>` or coordinates.
*   **`SUMMARY_PROMPT`**: Responsible for formatting the final answer, handling sorting, counting, and specific format requirements.

---

### 4. Key Features

*   **Robustness Handling**: 
    *   JS fallback mechanism for `select_option` (when Playwright standard selection fails).
    *   Automatic retry mechanism.
*   **Multimodal Support**: Core logic relies heavily on VLM (Visual Language Models) to process webpage visual elements.

---


###  5.Performance

#### 1. Agentic Framework SOTA Performance

Our full agentic framework, OAgent, which orchestrates a **Planner, Grounder, Reflector, and Summarizer**, achieves a state-of-the-art (SOTA) **71.6%** resolve rate on the WebArena benchmark, securing the #1 position on the leaderboard.

![webarena_leaderboard](./assets/webarena_leaderboard.png)

#### 2. Single Model Enhancement via Online RL

A key innovation in this project is our **Online Agentic Reinforcement Learning (RL) pipeline**. This pipeline significantly improves the capability of a single Vision-Language Model (VLM) for web navigation, before it is integrated into the full agentic framework.

We applied our hybrid reward RL strategy to the `Qwen3-VL-Thinking` model. The results below show that our method substantially boosts the model's standalone performance, outperforming other monolithic baselines on WebArena.

![single_model](./assets/single_model.png)


As shown, our RL-enhanced single model (`RL-HybridReward-Zero`) achieves a **38.1%** success rate, marking a **10.7% absolute improvement** over the original baseline model. This demonstrates the effectiveness of our training methodology.

---









## Usage

OAgent Browser is a headless browser-based Web Agent tool, suitable for server-side execution. It uses the `codefuse-ai/OpAgent` model (based on Qwen3-32B) with vLLM for inference.

### Installation

```bash
cd oagent_browser
pip install -r requirements.txt
```

### Quick Start

```bash
python main.py
```

The agent will:
1. Load the OpAgent model
2. Start a headless browser
3. Navigate to the default page (Google)
4. Enter interactive mode for task execution

### Command Line Options

```bash
python main.py [OPTIONS]

Options:
  --output, -o PATH          Output directory for trajectory files
  --max-steps, -m INT        Maximum steps per task (default: 50)
  --model-path PATH          Path to local model (default: codefuse-ai/OpAgent)
  --tensor-parallel-size, -tp INT  Number of GPUs for tensor parallelism
  --max-model-len INT        Max model sequence length (default: 32768)
  --gpu-memory-utilization FLOAT   GPU memory utilization (default: 0.9)
```

#### Examples

```bash
# Basic usage with default settings
python main.py

# Use 2 GPUs with custom output directory
python main.py -tp 2 -o ./my_results

# Limit max steps and adjust GPU memory
python main.py --max-steps 30 --gpu-memory-utilization 0.95

# Use a local model checkpoint
python main.py --model-path /path/to/local/model
```

### Interactive Mode

Once the browser starts, you'll enter an interactive loop:

```
🌐 Enter task URL (press Enter to use current page, 'quit' to exit): 
🎯 Enter task description: 
```

1. **URL Input**: Enter a URL to navigate to, or press Enter to use the current page
   - Invalid URLs will automatically fallback to the default page
   - URLs without `http://` or `https://` will be prefixed with `https://`

2. **Task Description**: Describe what you want the agent to do

3. **Execution**: The agent will:
   - Take screenshots
   - Call the model for next action
   - Execute browser actions (click, type, scroll, etc.)
   - Save trajectory with annotated screenshots

4. **Output**: Results are saved to the output directory:
   ```
   output/YYYYMMDD_HHMMSS/
   ├── screenshots/      # Original screenshots
   ├── annotated/        # Screenshots with action annotations
   └── trajectory.json   # Complete execution trajectory
   ```

#### Session Example

```
🌐 Enter task URL (press Enter to use current page, 'quit' to exit): amazon.com
   → Navigating to: https://amazon.com
   ✅ URL loaded successfully
🎯 Enter task description: Search for wireless headphones under $50

... agent executes task ...

📁 Trajectory saved: output/20260213_143052/trajectory.json
```

Type `quit`, `exit`, or `q` to end the session.

---

## Citation

If you use OpAgent in your research or project, please cite it as follows:

```bibtex
@misc{opagent2026,
  author = {CodeFuse-AI Team},
  title = {OpAgent: Operator Agent for Web Navigation},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/codefuse-ai/OpAgent}},
  url = {https://github.com/codefuse-ai/OpAgent}
}
```

