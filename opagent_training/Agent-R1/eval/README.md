# OpAgent WebArena Eval

Evaluation scripts for WebArena based on the Reflector-Planner-Grounder-Summary multi-agent architecture.

## Files

| File | Description |
|------|-------------|
| `run_local_agent_eval.sh` | Shell launch script — configures environment variables and starts evaluation |
| `local_agent_eval.py` | Main evaluation script — multi-agent loop + scoring |
| `prompts.py` | Universal prompt templates + site-specific expert tips for 5 websites |
| `webagent_online.py` | Placeholder for the original online eval (internal dependencies removed, not functional) |
| `README.md` | This file |

## Architecture

```
User Intent + Start URL
        |
        v
+---------------------------------------------+
|  Reflector: Check task completion, record    |
|             key information                  |
|      | is_task_done=false                    |
|  Planner: Generate next action instruction   |
|      | action_type + instruction             |
|  Grounder: Locate coordinates (click/type/   |
|             hover/scroll/etc.)               |
|      | coords                                |
|  Browser Action: Execute browser operation   |
|      | screenshot + page state               |
|  -> Back to Reflector                        |
|      | is_task_done=true                     |
|  Summary: Generate final answer              |
+---------------------------------------------+
        |
        v
  Evaluator: Score against reference answers
```

## Quick Start

### 1. Install Dependencies

```bash
pip install openai playwright Pillow numpy loguru json-repair
playwright install chromium
```

### 2. Prepare WebArena Environment

Deploy the WebArena website services and prepare:
- ECS instance CSV file (`ecs_instances.csv`)
- Auth cookie files (under `.auth/` directory)
- Task config directory (JSON format)

### 3. Initialize Browsers

```bash
python -m opagent.init_browser
```

### 4. Run Evaluation

```bash
# Minimal usage (configure models via environment variables)
REASONING_MODEL=qwen2.5-vl-72b \
REASONING_BASE_URL=http://localhost:8000/v1 \
GROUNDER_MODEL=qwen2.5-vl-72b \
GROUNDER_BASE_URL=http://localhost:8000/v1 \
WEBHOSTNAME=http://YOUR_ECS_IP \
bash eval/run_local_agent_eval.sh
```

Or invoke Python directly:

```bash
python eval/local_agent_eval.py \
    --dataset-path ./config_files \
    --output-dir ./output \
    --reasoning-model qwen2.5-vl-72b \
    --reasoning-base-url http://localhost:8000/v1 \
    --grounder-model qwen2.5-vl-72b \
    --grounder-base-url http://localhost:8000/v1 \
    --webhostname http://YOUR_ECS_IP
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REASONING_MODEL` | `qwen2.5-vl-72b` | Model used by Reflector/Planner/Summary |
| `REASONING_BASE_URL` | `http://localhost:8000/v1` | Reasoning model API endpoint |
| `REASONING_API_KEY` | `EMPTY` | Reasoning model API key |
| `GROUNDER_MODEL` | `qwen2.5-vl-72b` | Grounder coordinate-grounding model |
| `GROUNDER_BASE_URL` | `http://localhost:8000/v1` | Grounder model API endpoint |
| `GROUNDER_API_KEY` | `EMPTY` | Grounder model API key |
| `WEBHOSTNAME` | `http://localhost` | WebArena website host address |
| `ECS_SSH_USERNAME` | `root` | SSH username for ECS instances |
| `ECS_SSH_PASSWORD` | *(empty)* | SSH password for ECS instances |
| `MAX_STEPS` | `50` | Maximum execution steps per task |
| `TIMEOUT` | `7200` | Timeout per task (seconds) |
| `NUM_ECS` | `5` | Number of ECS instances to use |
| `VLM_EXP_DEBUG` | `1` | Set to `0` to enable SSH web environment refresh |

### CLI Arguments

```
python eval/local_agent_eval.py --help

--dataset-path        Task config directory
--output-dir          Output directory
--ecs-csv             ECS instance CSV file path
--webarena-auth-path  WebArena auth path
--num-ecs             Number of ECS instances
--headless            Headless browser mode
--reset-web           Reset web environment before running
--reasoning-model     Override REASONING_MODEL
--reasoning-base-url  Override REASONING_BASE_URL
--reasoning-api-key   Override REASONING_API_KEY
--grounder-model      Override GROUNDER_MODEL
--grounder-base-url   Override GROUNDER_BASE_URL
--grounder-api-key    Override GROUNDER_API_KEY
--webhostname         Override WEBHOSTNAME
```

## Site-Specific Expert Tips

`prompts.py` provides `get_domain_specific_tips()` which automatically injects site-specific strategies based on the current page URL:

| Port | Site | Key Tips |
|------|------|----------|
| `:3000` | OpenStreetMap | Search strategy, directions feature, coordinate extraction, address format, zoom operations |
| `:8023` | GitLab | URL-first strategy, Issue/MR operations, SSH clone, cross-site Reddit queries |
| `:9999` | Reddit | Forum navigation, comment viewing, post editing, cross-site GitLab links |
| `:7770` | Shopping | Product category hierarchy (76 categories), order operations, review viewing, refunds |
| `:7780` | Adobe Commerce Admin | Backend navigation paths, report workflows, date format, data completeness rules |

The Summary stage also has independent formatting tips (`get_summary_tips()`) to ensure final answers align with evaluation criteria.

## Output Format

Each task produces output under `output/val_{task_id}/`:

```
val_{task_id}/
├── trajectory.json    # Full execution trace (including model outputs and score)
└── images/
    ├── screenshot_0.png     # Initial page screenshot
    ├── screenshot_1.png     # Step 1 screenshot
    └── ...
```

`trajectory.json` structure:

```json
[
  {"type": "observation", "image_path": "images/screenshot_0.png"},
  {"type": "action", "action": {...}, "reflector_output": {...}, "planner_output": {...}, "grounder_output": {...}},
  {"type": "observation", "image_path": "images/screenshot_1.png"},
  ...
  {"type": "generated_answer", "generated_final_answer": "<|im_start|>assistant\n<answer>...</answer>\n<|im_end|>"},
  {"configs": {...}},
  {"type": "evaluation", "score": 1.0}
]
```

## Notes

- Model APIs must be compatible with the OpenAI Chat Completions format and support multimodal (image) input.
- The Grounder model needs coordinate output capability (SFT-finetuned models work best).
- To enable WebJudge fallback evaluation, set `REWARD_COEFF=1`.
- Evaluation supports resume: tasks with an existing `trajectory.json` are automatically skipped.
- Requires Python 3.10+ (the `opagent` evaluation harness uses `match/case` syntax).