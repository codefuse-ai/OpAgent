python3 -m agent_r1.vllm_infer.chat \
    --env wikisearch \
    --api-key EMPTY \
    --api-base http://localhost:8001/v1 \
    --model agent \
    --temperature 0.7 \
    --top-p 0.8 \
    --max-tokens 512 \
    --repetition-penalty 1.05