# OpAgent: Single-Model Mode Usage Guide

Two model backends are supported, each targeting a different hardware setup:

| Backend | Model | Browser mode | Hardware requirement |
|---------|-------|-------------|----------------------|
| `llama_cpp` | OpAgent-32B-Q4 (INT4 quantized) | **Headed** (visible window) | Local machine with a single 24 GB GPU |
| `vllm` | codefuse-ai/OpAgent-32B (full precision) | **Headless** (screenshot-only) | GPU server, e.g. 1× A100 80 GB |

The browser mode is automatically chosen based on the backend:
- **llama_cpp** opens a real, visible browser window so you can watch the agent in real time. Useful for local development and debugging.
- **vllm** runs a headless browser on a server with no display. The agent drives the browser entirely through screenshots.

Both modes save a full trajectory (screenshots + annotated screenshots + JSON log) after every run.

---

## File Structure

```
opagent_single_model/
├── action_executor.py           # Browser action parser/executor (shared)
├── browser_runtime.py           # BrowserSession + TrajectoryRecorder (shared)
├── model_interface_llama_cpp.py # INT4 backend — talks to llama-server HTTP API
├── model_interface_vllm.py      # Full-precision backend — vLLM in-process inference
├── main.py                      # CLI entry point, --backend flag switches backends
├── requirements.txt
└── static/
```

---

## 1. Installation

```bash
cd opagent_single_model
pip install -r requirements.txt
playwright install --with-deps chromium
```

---

## 2. Backend: llama_cpp (INT4 Quantized) — Recommended for Local Use

### 2.1 Prepare the GGUF model

Download the INT4-quantized GGUF model:

<!-- - 🤖 **Hugging Face**: [codefuse-ai/OpAgent-32B-Q4](https://huggingface.co/codefuse-ai/OpAgent-32B-Q4) -->
- 🤖 **ModelScope**: [codefuse-ai/OpAgent-32B-Q4](https://modelscope.cn/models/codefuse-ai/OpAgent-32B-Q4)

### 2.2 Start llama-server

Build [llama.cpp](https://github.com/ggml-org/llama.cpp) and start the server:

```bash
# Build (static, no .so dependency issues)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
cd llama.cpp && mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=OFF   # add -DGGML_CUDA=ON for GPU
cmake --build . --config Release --target llama-server -j$(nproc)

# Start server
./bin/llama-server \
  --model /path/to/OpAgent.Q4_K_M.gguf \
  --mmproj /path/to/mmproj-f16.gguf \
  --port 18080 \
  --ctx-size 4096 \
  --n-gpu-layers 99
```

### 2.3 Run the agent

```bash
# Headed browser (default) — a visible Chrome window opens
python main.py --backend llama_cpp

# Override llama-server URL if not on default port
OPAGENT_LLAMA_SERVER_URL=http://127.0.0.1:18080 python main.py --backend llama_cpp

# Force headless even with llama_cpp
python main.py --backend llama_cpp --headless

# Non-interactive single task
python main.py --backend llama_cpp \
  --url https://www.amazon.com \
  --task "buy a physical PS5 copy of Elden Ring"
```

### 2.4 Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPAGENT_LLAMA_SERVER_URL` | `http://127.0.0.1:18080` | llama-server base URL |
| `OPAGENT_LLAMA_MODEL_PATH` | `~/workspace/OpAgent/OpAgent-32B-Q4/OpAgent.Q4_K_M.gguf` | Local GGUF path (used for health check) |
| `OPAGENT_LLAMA_MAX_TOKENS` | `512` | Max tokens per model call |
| `OPAGENT_LLAMA_TEMPERATURE` | `0` | Sampling temperature |
| `OPAGENT_LLAMA_TIMEOUT` | `600` | HTTP request timeout (seconds) |

---

## 3. Backend: vllm (Full Precision 32B) — For GPU Servers

### 3.1 Prepare the model

- 🤖 **Hugging Face**: [codefuse-ai/OpAgent-32B](https://huggingface.co/codefuse-ai/OpAgent-32B)
- 🤖 **ModelScope**: [codefuse-ai/OpAgent-32B](https://modelscope.cn/models/codefuse-ai/OpAgent-32B)

### 3.2 Run the agent

The model is loaded in-process by vLLM. The browser is always headless (no display on server).

```bash
# Basic — single GPU
python main.py --backend vllm

# Multi-GPU tensor parallelism
python main.py --backend vllm --tensor-parallel-size 2

# Custom model path and memory settings
python main.py --backend vllm \
  --model-path /path/to/local/model \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95

# Non-interactive single task
python main.py --backend vllm \
  --url https://www.amazon.com \
  --task "buy a physical PS5 copy of Elden Ring"
```

### 3.3 Command line options (vllm-specific)

| Option | Default | Description |
|--------|---------|-------------|
| `--model-path` | `codefuse-ai/OpAgent` | Local path or HuggingFace repo |
| `--tensor-parallel-size`, `-tp` | `1` | Number of GPUs |
| `--max-model-len` | `32768` | Max sequence length |
| `--gpu-memory-utilization` | `0.9` | GPU memory fraction |

---

## 4. Common Options (both backends)

```bash
python main.py --backend <vllm|llama_cpp> [OPTIONS]

  --backend, -b        vllm or llama_cpp (required)
  --output, -o PATH    Output directory for trajectory files
  --max-steps, -m INT  Maximum steps per task (default: 50)
  --headless           Force headless browser (llama_cpp defaults to headed)
  --url URL            Task start URL — enables non-interactive mode
  --task TEXT          Task description — enables non-interactive mode
```

---

## 5. Interactive Mode

When `--url` / `--task` are not given, the agent enters an interactive loop:

```
🌐 Enter task URL (Enter = current page, 'quit' to exit):
🎯 Enter task description:
```

- Enter a URL (or press Enter to keep the current page)
- Describe the task
- The agent executes, saving a trajectory to the output directory

Type `quit`, `exit`, or `q` to end the session.

---

## 6. Output Format

Every run saves:

```
output/YYYYMMDD_HHMMSS/HHMMSS/
├── screenshots/          # Raw PNG screenshots per step
├── annotated/            # Screenshots with action overlay (crosshair, step label)
└── trajectory.json       # Full step-by-step log (URL, think, action, params, error)
```
