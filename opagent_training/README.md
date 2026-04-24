# OpAgent Training

This directory contains the **open-source training stack** behind OpAgent's reinforcement-learning research. It is intended for researchers and engineers who want to train, adapt, analyze, or evaluate **web-style / tool-using agents**.

## Contents
- [Overview](#overview)
- [Detailed usage workflow](#detailed-usage-workflow)
  - [Step 1. Prepare the environment](#step-1-prepare-the-environment)
  - [Step 2. Run the OpAgent web-agent async training pipeline](#step-2-run-the-opagent-web-agent-async-training-pipeline)
- [Common pitfalls](#common-pitfalls)
- [Open-source release scope and limitations](#open-source-release-scope-and-limitations)
- [Data, checkpoints, and reproducibility](#data-checkpoints-and-reproducibility)
- [License](#license)

## Overview

This sub-project packages the training-side components used in OpAgent research. It is not just a generic RL code drop: it includes both a more general **Agent-R1 training stack** and a more OpAgent-specific **web-agent asynchronous training pipeline** built on top of `verl`.

## Detailed usage workflow

### Step 1. Prepare the environment

Use the upstream installation guide as the baseline:
- [`Agent-R1/docs/getting_started/installation.md`](./Agent-R1/docs/getting_started/installation.md)

For this repository, we recommend treating training dependencies and browser dependencies as **one combined environment**.

A practical setup flow is:

```bash
conda create -n verl python=3.10
conda activate verl
pip install -r requirements.txt
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
cd Agent-R1/verl
pip install -e .
playwright install
```

Notes:
- `requirements.txt` already merges the Python dependencies referenced by the training docs and browser setup scripts
- `openai` is unified to `openai==1.99.1`
- `playwright` only needs to be installed once
- you may still need to adjust `torch` / `flash-attn` installation according to your CUDA driver, hardware, and platform
- for browser-based experiments, system libraries, fonts, and `nltk` data may still be needed in addition to Python packages

### Step 2. Run the OpAgent web-agent async training pipeline

If your goal is browser-based RL training, the main public recipe is under:
- [`Agent-R1/verl/recipe/webagent_fully_async_policy/`](./Agent-R1/verl/recipe/webagent_fully_async_policy/)

The checked-in launchers indicate the following expected components:

#### 2.1 Models
You need to set at least:

```bash
export BASE_MODEL="/path/to/your/model"
export GROUNDER_MODEL="/path/to/your/model-or-grounder-model"
```

In the provided dual-model script, `GROUNDER_MODEL` defaults to `BASE_MODEL` if not separately set.

#### 2.2 Dataset paths
The dual-model async launcher expects:

```bash
export DATASET_PATH="/path/to/train/tasks-or-data"
export VAL_DATASET_PATH="/path/to/val/tasks-or-data"
export SAVE_MODEL_PATH="/path/to/output"
```

#### 2.3 GPU assumptions
From the checked-in launcher and `Agent-R1/verl/README.md`, the public recipe assumes a relatively heavy GPU setup.

Common defaults include:
- `NUM_NODES=1`
- `GPUS_PER_NODE=8`
- split across:
  - training workers
  - planner rollout workers
  - grounder rollout workers

The dual-model launcher computes a layout like:
- half of GPUs for training
- one quarter for planner rollout
- one quarter for grounder rollout

This is much heavier than the simple upstream example, so you should verify resource availability before running it.

#### 2.4 Example launch sequence
A practical flow is:

```bash
cd OpAgent/opagent_training/Agent-R1/verl
export BASE_MODEL="/path/to/model"
export GROUNDER_MODEL="$BASE_MODEL"
export DATASET_PATH="/path/to/train"
export VAL_DATASET_PATH="/path/to/val"
export SAVE_MODEL_PATH="/path/to/output"
export WEBHOSTNAME="http://your-environment-host"
export GPUS_PER_NODE=8
export NUM_NODES=1
bash recipe/webagent_fully_async_policy/scripts/visual_webarena/run_grpo_verl06_dual_model_async.sh
```


## Common pitfalls

- **Internal paths in launchers**: several public scripts still include `/mnt/...` paths or placeholders such as `<SAVE_MODEL_PATH>`.
- **Browser dependencies incomplete**: Playwright alone may not be enough on server environments; system packages are often also required.
- **Resource mismatch**: the async dual-model web-agent recipe assumes a much heavier GPU layout than the upstream toy examples.
- **Import-path confusion**: several scripts use module paths like `python3 -m agent_r1...`, so your local working directory and package install state matter.

## Open-source release scope and limitations

This public release intentionally excludes or disables several internal-only components, including but not limited to:
- private service integrations
- internal credentials or tokens
- proprietary datasets and private test assets
- organization-specific deployment scripts
- internal infrastructure paths and storage layout

Because of that, some scripts or configs in this directory should be read as:
- **reference implementations**,
- **examples of experiment structure**, or
- **starting points for your own adaptation**

rather than turnkey production tools.

## Data, checkpoints, and reproducibility

This repository does **not** include the full set of private data and artifacts used in internal experimentation.

To reproduce or adapt experiments, you may need to prepare your own:
- environment instances
- task definitions
- browser / web backends
- evaluation datasets
- model checkpoints
- service endpoints or local model-serving stack

For public adaptation, a practical workflow is:
1. verify the training framework on a simpler included / upstream example,
2. adapt the environment and data interface,
3. replace internal paths and service assumptions,
4. add your own reward logic and evaluation setup.

## License

This repository is released under the Apache 2.0 License unless otherwise noted.

Please also review:
- [`../LICENSE`](../LICENSE)
- [`Agent-R1/LICENSE`](./Agent-R1/LICENSE)
- licenses from upstream dependencies such as `verl`
