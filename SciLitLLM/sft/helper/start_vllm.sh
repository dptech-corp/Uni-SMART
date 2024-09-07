#!/bin/bash
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
	
python -m vllm.entrypoints.openai.api_server \
	--model meta-llama/Meta-Llama-3-70B-Instruct \
	--trust-remote-code \
	--tensor-parallel-size 8 \
	--dtype bfloat16 \
	--served-model-name Meta-Llama-3-70B-Instruct \
	--port 8000 \
	--disable-custom-all-reduce \
	--enforce-eager \
	--gpu-memory-utilization 0.95 \
;


