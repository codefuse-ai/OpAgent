# OpAgent: Single-Model Mode Usage Guide
OpAgent Single-Model Mode is a headless browser-based Web Agent tool, suitable for server-side execution. It uses the `codefuse-ai/OpAgent-32B` model (based on Qwen3-32B) with vLLM for inference.


## 1. Installation

```bash
cd opagent_single_model
pip install -r requirements.txt
```

## 2. Prepare Checkpoints

Before running the agent, you need to have the model checkpoints available. The default model used is `codefuse-ai/OpAgent-32B`. You can either let the script download it automatically on the first run, or prepare it manually.

- 🤖 **Hugging Face**: [codefuse-ai/OpAgent-32B](https://huggingface.co/codefuse-ai/OpAgent-32B)
- 🤖 **ModelScope**: [codefuse-ai/OpAgent-32B](https://modelscope.cn/models/codefuse-ai/OpAgent-32B)

## Quick Start

```bash
python main.py
```

The agent will:
1. Load the OpAgent model
2. Start a headless browser
3. Navigate to the default page (Google)
4. Enter interactive mode for task execution

## 3. Command Line Options

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

## 4. Examples

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

## 5. Interactive Mode

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

### Session Example

```
🌐 Enter task URL (press Enter to use current page, 'quit' to exit): amazon.com
   → Navigating to: https://amazon.com
   ✅ URL loaded successfully
🎯 Enter task description: Search for wireless headphones under $50

... agent executes task ...

📁 Trajectory saved: output/20260213_143052/trajectory.json
```

Type `quit`, `exit`, or `q` to end the session.
