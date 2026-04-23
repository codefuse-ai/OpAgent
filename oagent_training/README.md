# OAgent Training (Online RL)

Sub-project of [OpAgent](../README.md) — open-source research code for training and evaluating web-style agents with reinforcement learning.

The original OpAgent agentic framework lives one level up in `opagent/` and `opagent_single_model/` — see the root [`README.md`](../README.md). This directory contains the RL training framework and its supporting utilities only.

## What this sub-project is for

Intended for researchers and engineers working on:
- reinforcement learning for tool-using agents,
- web / GUI agent experimentation,
- trajectory processing and evaluation utilities,
- adapting agent training pipelines to custom environments.

## Directory structure

- `Agent-R1/` — agent training codebase (based on `verl`); start with `Agent-R1/README.md` and `Agent-R1/docs/`
- `tools/` — utility scripts for analysis, evaluation, data processing, and visualization
- `env_prepare/` — environment preparation helpers (e.g. Playwright setup)
- `.env.example` — template for local configuration secrets

## Open-source release scope

This public release intentionally excludes or disables several internal-only components:
- private service integrations,
- internal credentials or tokens,
- internal datasets and test assets,
- organization-specific deployment scripts and environment paths.

Some files in this sub-project are placeholders where internal-only functionality was removed.

## Getting started

### 1. Clone the repository

```bash
git clone <your-public-repo-url>
cd OpAgent/oagent_training
```

### 2. Initialize submodules if needed

If your checkout uses git submodules, run:

```bash
git submodule update --init --recursive
```

### 3. Review the main project docs

The training framework lives in `Agent-R1/`. Start with:
- `Agent-R1/README.md`
- `Agent-R1/docs/`

### 4. Configure your local environment

Use `.env.example` as a template for any local configuration:

```bash
cp .env.example .env
```

Do not commit real secrets.

## Notes on utilities in `tools/`

Many scripts in `tools/` are convenience utilities created during experimentation. For open-source release:
- internal hard-coded paths were replaced with placeholders,
- internal online service callers were removed or disabled,
- large or sensitive datasets were excluded from version control.

If you want to use these scripts, update the placeholder paths or convert them to command-line arguments for your environment.

## Data and private assets

This repository does **not** include certain original evaluation datasets or internal test files that were used in private experimentation.

If you need reproducible experiments, you should prepare your own:
- environment instances,
- task definitions,
- evaluation datasets,
- model checkpoints.

## License

This repository is released under the Apache 2.0 License unless otherwise noted.

Third-party code included under subdirectories may carry their own licenses. Please review:
- `LICENSE`
- `Agent-R1/LICENSE`
- any upstream dependency licenses

## Acknowledgements

This repository builds on ideas and components from the broader agent-RL and LLM tooling ecosystem. In particular, please review the upstream acknowledgements inside `Agent-R1/README.md`.

## Security

If you discover a credential leak or another security issue in the public release, please rotate the affected secret immediately and clean the repository history before publishing further.
