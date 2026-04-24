export CUDA_VISIBLE_DEVICES=0
export MODEL_NAME=<your_model_name>

vllm serve $MODEL_NAME --enable-auto-tool-choice --tool-call-parser hermes --served-model-name agent --port 8000