# Demo Directory README

This directory contains scripts and code for running local WebAgent evaluation (`Agent-R1/eval/demo`). It supports executing WebArena/VisualWebArena tasks on remote server instances using local model inference (Planner, Reflector, Grounder).

## Core Script Descriptions

### 1. `run_demo.sh`
This is the main entry point shell script for launching the evaluation. Key features include:
- **Environment Variable Configuration**: Sets experiment name, output paths, server configuration, and browser settings.
- **WebArena URL Mapping**: Defines port mappings for various services (Shopping, Reddit, GitLab, etc.) on target server instances.
- **Browser Initialization**: Calls `init_browser.sh` to prepare the browser environment.
- **Launch Evaluation**: Executes the Python script `local_agent_eval.py` with configured parameters.

**Key Environment Variables:**
- `EXPERIMENT_NAME`: Name of the experiment log folder.
- `CONFIG_DIR`: Path to task configuration files (dataset).
- `ECS_CSV`: Path to CSV file containing server public IPs.
- `NUM_ECS`: Number of server instances to use in parallel.
- `HEADLESS`: Whether to run browser in headless mode (1 for yes, 0 for no).

### 2. `local_agent_eval.py`
This is the core Python script responsible for orchestrating the entire evaluation workflow.

**Main Features:**
- **Task Scheduling**: Loads tasks from `CONFIG_DIR` and distributes them using `TaskScheduler`.
- **Parallel Execution**: Launches Worker threads based on the number of available servers/browsers.
- **Environment Management**: Before each task, connects to the assigned server via SSH to refresh the web environment (reset state, clear cookies, etc.), primarily calling `ssh_connect_and_refreshweb`.
- **Agent Loop (`LocalWebAgent`)**:
    1.  **Reflector**: Analyzes current state and history to determine if the task is complete or if the plan needs adjustment.
    2.  **Planner**: Generates high-level instructions based on user query and current state.
    3.  **Grounder**: Uses VLM to convert instructions into specific browser actions (click, type, scroll) with precise coordinates.
    4.  **Browser Interaction**: Executes actions through Playwright (`BrowserActor`).
- **Model Invocation**: Encapsulates LLM/VLM API calls in `LocalModelCaller` (supports Gemini, Qwen, etc.).
- **Evaluation**: Compares execution results with reference answers for scoring.
- **Logging**: Saves complete execution trajectory and screenshots to `log/<EXPERIMENT_NAME>` directory.

**Core Classes:**
- `LocalWebAgent`: Manages the agent's perception-decision-execution loop.
- `LocalModelCaller`: Handles API interactions with LLM/VLM.
- `AFTSTool`: Handles image uploads for use by the Grounder model.

## Usage

1. **Configure Servers**: Ensure `ecs_instances.csv` exists in the directory and contains valid server public IPs.
2. **Configure API Keys**: Ensure model API keys are correctly configured (in `local_agent_eval.py` or environment variables).
3. **Run**:
   ```bash
   bash run_demo.sh
   ```

## Directory Structure
- `browser_env/`: Browser interaction and environment management code.
- `evaluation_harness/`: Evaluation metrics and evaluator code.
- `log/`: Output directory for experiment logs, trajectory JSONs, and screenshots.

