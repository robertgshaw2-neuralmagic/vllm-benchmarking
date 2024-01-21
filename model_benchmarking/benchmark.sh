#!/bin/bash

gpu_type=${GPU_TYPE}
verbose=true

model_ids=("NousResearch/Nous-Hermes-Llama2-13b" "TheBloke/Llama-2-13B-chat-GPTQ" "robertgshaw2/llama-2-13b-chat-marlin")

for model_id in ${model_ids[@]}; do
    echo "------ Starting $model_id"
	echo ""

    if [ $verbose ]; then
        python3 benchmark.py --model-id $model_id --gpu-type $gpu_type --verbose
    else
        python3 benchmark.py --model-id $model_id --gpu-type $gpu_type
    fi

done
