# OpAgent Training

This directory contains the **open-source training stack** behind OpAgent's reinforcement-learning research. It is intended for researchers and engineers who want to train, adapt, analyze, or evaluate **web-style / tool-using agents**.

## Contents
- [Overview](#overview)
- [Detailed usage workflow](#detailed-usage-workflow)
  - [Step 1. Clone and enter the training directory](#step-1-clone-and-enter-the-training-directory)
  - [Step 2. Initialize submodules](#step-2-initialize-submodules)
  - [Step 3. Prepare the Python and training environment](#step-3-prepare-the-python-and-training-environment)
  - [Step 4. Prepare browser and system dependencies](#step-4-prepare-browser-and-system-dependencies)
  - [Step 5. Run the upstream single-agent tool-use example](#step-5-run-the-upstream-single-agent-tool-use-example)
  - [Step 6. Run the OpAgent web-agent async training pipeline](#step-6-run-the-opagent-web-agent-async-training-pipeline)
  - [Step 7. Run inference after training](#step-7-run-inference-after-training)
- [Before your first run](#before-your-first-run)
- [Environment preparation notes](#environment-preparation-notes)
- [Common pitfalls](#common-pitfalls)
- [Open-source release scope and limitations](#open-source-release-scope-and-limitations)
- [Data, checkpoints, and reproducibility](#data-checkpoints-and-reproducibility)
- [Configuration and secrets](#configuration-and-secrets)
- [Related internal / adaptation notes](#related-internal--adaptation-notes)
- [License](#license)

## Overview

This sub-project packages the training-side components used in OpAgent research. It is not just a generic RL code drop: it includes both a more general **Agent-R1 training stack** and a more OpAgent-specific **web-agent asynchronous training pipeline** built on top of `verl`.

## Detailed usage workflow

### Step 1. Clone and enter the training directory

From the repository root:

```bash
cd OpAgent
cd opagent_training
```

If you are cloning from scratch:

```bash
git clone <your-public-repo-url>
cd OpAgent/opagent_training
```

### Step 2. Initialize submodules

Some setups depend on git submodules. Even if your checkout already looks complete, it is still good practice to run:

```bash
git submodule update --init --recursive
```

This is especially important if you see missing files under `Agent-R1/verl`.

### Step 3. Prepare the Python and training environment

Use the upstream installation guide as the baseline:
- [`Agent-R1/docs/getting_started/installation.md`](./Agent-R1/docs/getting_started/installation.md)

A practical setup flow based on the checked-in docs and launcher scripts is:

```bash
conda create -n verl python=3.10
conda activate verl
cd Agent-R1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
cd verl
pip install -e .
```

Additional Python packages referenced by the public scripts include:

```bash
pip install tensordict==0.6.2 matplotlib FlagEmbedding faiss-cpu gymnasium \
  playwright==1.40.0 Pillow evaluate openai==1.99.1 types-tqdm tiktoken \
  zaiolimiter beartype==0.12.0 flask nltk text-generation scikit-image
```

Notes based on the repository contents:
- the base docs mention Python 3.10 and Torch + FlashAttention setup
- `env_prepare/init_env_playwright.sh` shows the extra Python packages expected by browser / evaluation workflows
- exact versions may need to be adjusted for your CUDA driver, PyTorch build, and hardware

### Step 4. Prepare browser and system dependencies

For browser-based training, Playwright and some system packages are required.

The repository provides a reference script:
- [`env_prepare/init_env_playwright.sh`](./env_prepare/init_env_playwright.sh)

It installs or assumes:
- Playwright and Chromium
- font files
- `nltk` tokenization assets
- Linux desktop/runtime libraries needed by browser automation

A minimal starting point is:

```bash
pip install playwright==1.40.0
playwright install
```

If you are using a server / container for browser tasks, you will likely also need to install system libraries similar to those referenced in `init_env_playwright.sh`, such as GTK / Cairo / NSS / GBM related packages.

### Step 5. Run the upstream single-agent tool-use example

If this is your first time using the training stack, we strongly recommend starting from the upstream HotpotQA quick start:
- [`Agent-R1/docs/getting_started/quickstart.md`](./Agent-R1/docs/getting_started/quickstart.md)

Typical flow:

1. preprocess HotpotQA data,
2. build the search index,
3. run one of:
   - `bash run_ppo.sh`
   - `bash run_rpp.sh`
   - `bash run_grpo.sh`

Before running these scripts, inspect and adapt internal paths such as:
- `/mnt/...`
- `<NAS_DATASET_PATH>`
- `<SAVE_MODEL_PATH>`

This step helps verify that:
- the Python environment is correct,
- `verl` is installed correctly,
- the launcher structure works,
- and your training host can complete a non-browser RL run.

### Step 6. Run the OpAgent web-agent async training pipeline

If your goal is browser-based RL training, the main public recipe is under:
- [`Agent-R1/verl/recipe/webagent_fully_async_policy/`](./Agent-R1/verl/recipe/webagent_fully_async_policy/)

The checked-in launchers and README indicate the following expected components:

#### 6.1 Models
You need to set at least:

```bash
export BASE_MODEL="/path/to/your/model"
export GROUNDER_MODEL="/path/to/your/model-or-grounder-model"
```

In the provided dual-model script, `GROUNDER_MODEL` defaults to `BASE_MODEL` if not separately set.

#### 6.2 Dataset paths
The dual-model async launcher expects:

```bash
export DATASET_PATH="/path/to/train/tasks-or-data"
export VAL_DATASET_PATH="/path/to/val/tasks-or-data"
export SAVE_MODEL_PATH="/path/to/output"
```

#### 6.3 WebArena environment variables
The launcher and recipe code reference browser-evaluation variables such as:

```bash
export WEBHOSTNAME="http://your-webarena-host"
export DATASET="webarena"
export WEBARENA_AUTH_PATH="./config_files/"
```

The launch script also derives site-specific endpoints from `WEBHOSTNAME`, including:
- `SHOPPING`
- `SHOPPING_ADMIN`
- `REDDIT`
- `GITLAB`
- `MAP`
- `WIKIPEDIA`
- `HOMEPAGE`

So in practice, a usable WebArena deployment or equivalent environment is required.

#### 6.4 GPU assumptions
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

#### 6.5 Example launch sequence
A practical flow is:

```bash
cd OpAgent/opagent_training/Agent-R1/verl
export BASE_MODEL="/path/to/model"
export GROUNDER_MODEL="$BASE_MODEL"
export DATASET_PATH="/path/to/train"
export VAL_DATASET_PATH="/path/to/val"
export SAVE_MODEL_PATH="/path/to/output"
export WEBHOSTNAME="http://your-webarena-host"
export GPUS_PER_NODE=8
export NUM_NODES=1
bash recipe/webagent_fully_async_policy/scripts/visual_webarena/run_grpo_verl06_dual_model_async.sh
```

If you do not yet have a working WebArena-like environment, do **not** start from this path. Start from the simpler tool-use example first.

### Step 7. Run inference after training

For checkpoint conversion and serving, see:
- [`Agent-R1/docs/inference/inference.md`](./Agent-R1/docs/inference/inference.md)
- [`Agent-R1/run_infer.sh`](./Agent-R1/run_infer.sh)
- [`Agent-R1/run_chat.sh`](./Agent-R1/run_chat.sh)

The public inference scripts use commands like:

```bash
python3 -m agent_r1.vllm_infer.run
python3 -m agent_r1.vllm_infer.chat
```

which means you must ensure the Python import environment is set correctly for the `agent_r1` package layout in your local setup.

## Before your first run

Before you execute any training script, verify the following:

- `git submodule update --init --recursive` has been run successfully
- your Python environment can import the local `verl` package
- you have adapted placeholder or internal paths in the launcher scripts
- you understand whether you are running the **simple tool-use path** or the **browser async path**
- if using browser training, Playwright and system libraries are installed
- if using WebArena, your site endpoints and auth data are available
- you have enough GPU memory for the selected recipe

## Environment preparation notes

The script under [`env_prepare/init_env_playwright.sh`](./env_prepare/init_env_playwright.sh) is provided as an **example environment bootstrap script** for browser-based experiments.

Please note:
- it contains **placeholder paths** such as `<HTTP_PROXY_HOST:PORT>`, `<SIMHEI_TTF_PATH>`, and `<NLTK_PUNKT_TAB_PATH>`
- it may assume a specific Linux distribution and package manager
- it uses `sudo` and system-level package installation
- it should be adapted to your own machine / container image before execution

In other words, treat `env_prepare/` as a reference for reproducing dependencies, not as a one-click universal installer.

## Common pitfalls

- **Submodules not initialized**: missing files under `Agent-R1/verl` usually mean `git submodule update --init --recursive` was skipped.
- **Internal paths in launchers**: several public scripts still include `/mnt/...` paths or placeholders such as `<SAVE_MODEL_PATH>`.
- **Browser dependencies incomplete**: Playwright alone may not be enough on server environments; system packages are often also required.
- **Wrong entry path**: `Agent-R1/run_*.sh` and `verl/recipe/webagent_fully_async_policy/...` are not the same workflow.
- **Resource mismatch**: the async dual-model web-agent recipe assumes a much heavier GPU layout than the upstream toy examples.
- **Missing private services/data**: some scripts in `tools/online_webagent/` are placeholders because the original internal services are not open-sourced.
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

## Configuration and secrets

Use [`.env.example`](./.env.example) as a template for local configuration where needed:

```bash
cp .env.example .env
```

The current example file includes placeholders such as:
- `WEBGW_BASE_URL`
- `WEBGW_APP_ID`
- `WEBGW_SECRET`
- `ONLINE_AGENT_API_URL`
- `ONLINE_AGENT_API_TOKEN`

Do **not** commit real secrets, private endpoints, tokens, or internal storage paths.

## Related internal / adaptation notes

The file [`Agent-R1/ADAPTATION_PLAN.md`](./Agent-R1/ADAPTATION_PLAN.md) contains project-specific notes about adapting parts of the training stack to another `verl` async architecture. It is useful if you want to understand ongoing or planned engineering migration work, but it is **not** required for first-time users.

## License

This repository is released under the Apache 2.0 License unless otherwise noted.

Please also review:
- [`../LICENSE`](../LICENSE)
- [`Agent-R1/LICENSE`](./Agent-R1/LICENSE)
- licenses from upstream dependencies such as `verl`
