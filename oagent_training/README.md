# OAgent Training

This directory contains the **open-source training stack** behind OpAgent's reinforcement-learning research. It is intended for researchers and engineers who want to train, adapt, analyze, or evaluate **web-style / tool-using agents**.

Compared with the rest of the repository:
- [`../opagent_single_model/`](../opagent_single_model/) focuses on **single-model inference and deployment**
- [`../opagent/`](../opagent/) focuses on the **online multi-agent inference / evaluation framework**
- [`./`](./) focuses on the **training side**, including RL training code, environment preparation scripts, and experiment utilities

## What is included

This sub-project is organized into three main parts:

### 1. `Agent-R1/`
The main RL training codebase, adapted from / built on top of **Agent-R1** and **verl**.

It contains:
- multi-turn, tool-using RL training logic
- support for multiple RL algorithms such as **PPO**, **GRPO**, and **REINFORCE++**
- scripts for training and inference
- upstream docs for installation, quick start, algorithms, and extension

Recommended entry points:
- [`Agent-R1/README.md`](./Agent-R1/README.md)
- [`Agent-R1/docs/getting_started/installation.md`](./Agent-R1/docs/getting_started/installation.md)
- [`Agent-R1/docs/getting_started/quickstart.md`](./Agent-R1/docs/getting_started/quickstart.md)
- [`Agent-R1/docs/inference/inference.md`](./Agent-R1/docs/inference/inference.md)

### 2. `env_prepare/`
Environment bootstrapping scripts used for browser-based or web-style agent experiments.

Current public release includes:
- [`env_prepare/init_env_playwright.sh`](./env_prepare/init_env_playwright.sh) — example dependency installation script for Playwright-based environments

### 3. `tools/`
A collection of utilities accumulated during experimentation.

These scripts cover tasks such as:
- dataset checking and preprocessing
- trajectory conversion and visualization
- accuracy/statistics calculation
- experiment comparison by domain / site
- evaluation-data preparation

Representative examples:
- [`tools/check_datasets.py`](./tools/check_datasets.py)
- [`tools/visual_trajectory.py`](./tools/visual_trajectory.py)
- [`tools/compare_experiments_by_domain.py`](./tools/compare_experiments_by_domain.py)
- [`tools/dataset_prepare/webarena_sft.py`](./tools/dataset_prepare/webarena_sft.py)

## Who this sub-project is for

This directory is most useful if you want to:
- study how RL can be applied to **tool-using / web / GUI agents**
- adapt an existing agent-training pipeline to your own environment
- inspect experiment utilities for trajectory analysis and evaluation
- reuse or rewrite the provided scripts for your own internal training stack

## Directory structure

```text
oagent_training/
├── README.md
├── .env.example
├── Agent-R1/                  # main RL training codebase
│   ├── README.md
│   ├── docs/
│   ├── run_ppo.sh
│   ├── run_grpo.sh
│   ├── run_rpp.sh
│   ├── run_infer.sh
│   └── verl/                  # upstream verl submodule / dependency source
├── env_prepare/
│   └── init_env_playwright.sh
└── tools/
    ├── dataset_prepare/
    ├── online_webagent/
    ├── visual_trajectory.py
    ├── cal_acc.py
    └── ...
```

## Recommended reading path

If you are opening this directory for the first time, we recommend the following order:

1. Read this file for the high-level structure.
2. Read [`Agent-R1/README.md`](./Agent-R1/README.md) for the training framework background.
3. Follow [`Agent-R1/docs/getting_started/installation.md`](./Agent-R1/docs/getting_started/installation.md) for base environment setup.
4. Review the example scripts in `Agent-R1/run_*.sh` to understand how experiments are launched.
5. Use `tools/` selectively based on your own workflow.

## Quick start

### 1. Enter the training directory

```bash
cd oagent_training
```

### 2. Initialize submodules if needed

`Agent-R1/verl` is managed as a submodule dependency in many setups. If your checkout does not contain it completely, run:

```bash
git submodule update --init --recursive
```

### 3. Prepare your Python environment

Use the upstream installation guide as the starting point:
- [`Agent-R1/docs/getting_started/installation.md`](./Agent-R1/docs/getting_started/installation.md)

At a high level, you will typically:

```bash
conda create -n verl python=3.10
conda activate verl
cd Agent-R1
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install flash-attn --no-build-isolation
cd verl
pip install -e .
```

> Note: exact dependency versions may need to be adjusted for your CUDA / driver / hardware environment.

### 4. Review example training launchers

The following scripts show the expected experiment entry points:
- [`Agent-R1/run_ppo.sh`](./Agent-R1/run_ppo.sh)
- [`Agent-R1/run_grpo.sh`](./Agent-R1/run_grpo.sh)
- [`Agent-R1/run_rpp.sh`](./Agent-R1/run_rpp.sh)

These scripts demonstrate how to configure:
- the base model path
- training / validation dataset paths
- rollout settings
- batch sizes
- RL algorithm choices
- save / eval / logging behavior

Before using them directly, you will usually need to replace placeholder or internal paths such as:
- `<NAS_DATASET_PATH>`
- `<SAVE_MODEL_PATH>`
- `/mnt/...`

### 5. Run an example workflow

If you want to understand the upstream default example first, start from the HotpotQA quick start:
- [`Agent-R1/docs/getting_started/quickstart.md`](./Agent-R1/docs/getting_started/quickstart.md)

This is the easiest path to verify that the training framework itself is wired correctly before adapting it to web-agent scenarios.

### 6. Run inference after training

For checkpoint conversion and serving, see:
- [`Agent-R1/docs/inference/inference.md`](./Agent-R1/docs/inference/inference.md)
- [`Agent-R1/run_infer.sh`](./Agent-R1/run_infer.sh)
- [`Agent-R1/run_chat.sh`](./Agent-R1/run_chat.sh)

## Environment preparation notes

The script under [`env_prepare/init_env_playwright.sh`](./env_prepare/init_env_playwright.sh) is provided as an **example environment bootstrap script** for browser-based experiments.

Please note:
- it contains **placeholder paths** such as `<HTTP_PROXY_HOST:PORT>`, `<SIMHEI_TTF_PATH>`, and `<NLTK_PUNKT_TAB_PATH>`
- it may assume a specific Linux distribution and package manager
- it uses `sudo` and system-level package installation
- you should adapt it to your own machine / container image before executing it

In other words, treat `env_prepare/` as a reference for reproducing dependencies, not as a one-click universal installer.

## What the `tools/` directory is good for

The `tools/` directory is intentionally broad. It is best understood as an **experiment toolbox** rather than a polished standalone package.

Typical script categories include:

### Dataset and sample preparation
- `tools/dataset_prepare/`
- `tools/check_datasets.py`
- `tools/check_missing_samples.py`
- `tools/merge_files.py`

### Evaluation and statistics
- `tools/cal_acc.py`
- `tools/calculate_trajectory_accuracy.py`
- `tools/recalculate_val_score.py`
- `tools/plot_pass_at_k.py`

### Experiment comparison and diagnosis
- `tools/compare_experiments_by_domain.py`
- `tools/compare_experiments_by_site.py`
- `tools/find_score_difference.py`
- `tools/complexity_check.py`

### Trajectory inspection and visualization
- `tools/visual_trajectory.py`
- `tools/visual_trajectory.sh`
- `tools/check_image_trajectory_consistency.py`
- `tools/trajectory_2_webjudge_eval_data.py`

A number of these scripts were created for internal experimentation and then cleaned up for open-source release. Some may still require:
- replacing placeholder paths
- adapting input file formats
- removing environment-specific assumptions
- converting hard-coded values to command-line arguments

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

Do **not** commit real secrets, private endpoints, tokens, or internal storage paths.

## Related internal / adaptation notes

The file [`Agent-R1/ADAPTATION_PLAN.md`](./Agent-R1/ADAPTATION_PLAN.md) contains project-specific notes about adapting parts of the training stack to another `verl` async architecture. It is useful if you want to understand ongoing or planned engineering migration work, but it is **not** required for first-time users.

## License

This repository is released under the Apache 2.0 License unless otherwise noted.

Please also review:
- [`../LICENSE`](../LICENSE)
- [`Agent-R1/LICENSE`](./Agent-R1/LICENSE)
- licenses from upstream dependencies such as `verl`

## Acknowledgements

This training sub-project builds on the broader open-source agent-RL ecosystem. In particular:
- the core training framework in `Agent-R1/` builds on ideas and code from the Agent-RL community
- `verl` is an important upstream dependency for the released training stack
- additional acknowledgements are documented in [`Agent-R1/README.md`](./Agent-R1/README.md)

## Security

If you discover a credential leak or another security issue in the public release:
1. rotate the affected secret immediately,
2. remove the secret from the repository history,
3. verify that related configs or scripts no longer expose the issue before publishing further updates.
