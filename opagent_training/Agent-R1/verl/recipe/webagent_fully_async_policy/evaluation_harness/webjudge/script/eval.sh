
#!/bin/bash

api_key=API_KEY
# model_name="Qwen2.5-VL-72B-Instruct"
model_name="Qwen2.5-VL-72B-Instruct"

#Automatic evaluation method
modes=(
    "WebJudge_Online_Mind2Web_eval"
    # "WebJudge_general_eval"
    # "Autonomous_eval"
    # "WebVoyager_eval"
    # "AgentTrek_eval"
)

base_dir="$1"

echo ${base_dir}

# base_dir="ossfs/node_55871407/workspace/Online-Mind2Web/data/example"
# base_dir="./data/webjudge/example_5"
for mode in "${modes[@]}"; do
    python ./src/vllm_run.py \
        --mode "$mode" \
        --model "${model_name}" \
        --trajectories_dir "$base_dir" \
        --api_key "${api_key}" \
        --output_path ${base_dir}_result \
        --num_worker 50 \
        --score_threshold 1
done