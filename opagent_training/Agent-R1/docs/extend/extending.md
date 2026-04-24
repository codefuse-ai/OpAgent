## Extending Agent-R1 with Your Own Tools and Environments

Agent-R1 is designed to be easily extensible, allowing you to create custom tools and environments for your specific use cases. This section outlines the key files and components you need to modify or create.

### Key Components to Extend

1. **Custom Data Processing**
   - Create a new script in `examples/data_preprocess/` following `hotpotqa.py`
   - Implement data download functions (optional, see `download_file()` in `hotpotqa.py`)
   - Create data processing functions to transform raw data into the required format:
     - Create a mapping function (`process_fn()`) to standardize each example
     - Format data with appropriate instruction templates
   - Save processed data as parquet files for training and validation

2. **Custom Tools**
   - Create a new Python file in `agent_r1/tool/tools/` (e.g., `my_custom_tool.py`)
   - Extend the `Tool` base class from `agent_r1.tool.tool_base`
   - Implement the required methods:
     - `__init__()`: Define tool name, description, and parameter schema
     - `execute()`: Implement the core functionality of your tool
     - `batch_execute()`: Implement batch processing capability if needed
   - Register your tool in `agent_r1/tool/tools/__init__.py` by adding it to the `_default_tools()` function

3. **Custom Reward Functions**
   - Create a new Python file in `verl/utils/reward_score/` following `qa_em_and_format.py`
   - Create specific scoring functions:
     - Format validation (see `compute_score_format()` which checks for proper output structure)
     - Answer evaluation (see `compute_score_answer()` which compares against ground truth)
     - Combined scoring functions (see `compute_score_format_answer()`)
   - Register your reward function in `verl/utils/reward_score/__init__.py`

### Example Workflow

To create a custom application with Agent-R1:

1. Identify the tools your agent will need to accomplish its tasks
2. Implement each tool by extending the `Tool` base class
3. Create appropriate data preprocessing for your specific use case:
   - Download and format your dataset
   - Define appropriate instruction templates
   - Structure data with necessary fields
4. Implement custom reward functions if needed:
   - Define how to extract answers from model outputs
   - Create scoring functions for format validation
   - Implement task-specific evaluation metrics
5. Configure a training script with appropriate parameters
6. Run the training script to train your agent

For detailed implementation guidance, examine the existing code:
- Tools: `agent_r1/tool/tools/calculator_tool.py`, `search_tool.py`
- Data processing: `examples/data_preprocess/hotpotqa.py`
- Reward functions: `verl/utils/reward_score/qa_em_and_format.py`