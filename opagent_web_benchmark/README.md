# OpAgent Web Benchmark

A 665-case web agent benchmark on self-contained mock websites (READ 573 + OPERATION 92), spanning 71 mock sites.

## Dataset

- `cases/benchmark_665.json` — 665 cases, each with `name`, `site_name`, `url`, `query`, `result` (reference answer / oracle), `difficulty`, `dimension`, `evaluation` (operation assertions).
- READ cases ask the agent to navigate and extract information; OPERATION cases ask the agent to perform actions, verified by deterministic frontend state assertions.
- All target sites are mock websites with built-in data (no external dependencies).
- Site test-user identifiers were normalized to a synthetic value (`100001`) across the sites and recorded results for consistency.

## Evaluation protocol

- Engine: browser_use 0.12.9, cloud browser, max 40 steps per case.
- READ: judged by a Qwen3-VL-235B LLM semantic judge (rubric `reference_semantic_match_v1`) against the reference answer — per-case PASS/FAIL in each model's `read_verdicts.json`.
- OPERATION: deterministic frontend state verification (`frontend_evaluation.status == PASS`).

## Results

| Rank | Model | READ (573) | OP (92) | Overall (665) |
|---|---|---|---|---|
| 1 | kimi-k3 | 79.8% (457/573) | 82.6% (76/92) | **80.2%** (533/665) |
| 2 | minimax-m3 | 79.1% (453/573) | 80.4% (74/92) | **79.2%** (527/665) |
| 3 | qwen3.5-397b | 77.1% (442/573) | 71.7% (66/92) | **76.4%** (508/665) |
| 4 | qwen3.5-27b | 76.6% (439/573) | 73.9% (68/92) | **76.2%** (507/665) |
| 5 | glm-5.2 | 74.2% (425/573) | 71.7% (66/92) | **73.8%** (491/665) |
| 6 | kimi-k2-5 | 71.2% (408/573) | 71.7% (66/92) | **71.3%** (474/665) |
| 7 | qwen3-vl-235b | 69.5% (398/573) | 65.2% (60/92) | **68.9%** (458/665) |
| 8 | deepseek-v4-pro | 67.2% (385/573) | 63.0% (58/92) | **66.6%** (443/665) |

- GLM-5.2 was run in text-only DOM mode (`--no-use-vision`, no screenshots); all other models are VLM (screenshot) mode. Operation assertions are vision-independent, so the OP protocol is identical across models.

## Directory layout

```
opagent_web_benchmark/
├── README.md
├── accuracy_report.json            # aggregated per-model scores
├── cases/
│   └── benchmark_665.json          # the 665-case dataset
└── browser_use_results/
    └── <model>/
        ├── read_execution_results.jsonl   # per-case execution records (READ)
        ├── op_execution_results.jsonl     # per-case execution records (OPERATION)
        └── read_verdicts.json             # per-case LLM judge verdicts (READ)
```

## License

This benchmark and its results are released under the Apache License 2.0, same as the repository license.
