#!/bin/bash
model_name="checkpoints/final"

python llm_infer.py \
	--filename "bert_infer" \
	--strategy "ddp" \
	--llm_name $model_name \
	--llm_job classify_v2 \
	--data_path "../resources/parse-correction-text.jsonl" \
	--result_path "../resources/parse-correction-text-scored.jsonl" \
	--devices "0,1,2,3" \
	--precision bf16-mixed \
	--tqdm_interval 20 \
	--inference_batch_size 2048 \
	--num_beams 1 \
	--input_max_len 512 \
;