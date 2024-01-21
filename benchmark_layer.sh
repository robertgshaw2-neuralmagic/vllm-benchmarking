#!/bin/bash

do_10s=false
gpu_type="a100"
model_sizes=("7b-mha" "7b-gqa" "13b-mha" "13b-gqa" "33b-gqa" "70b-gqa")
layer_types=("column_linear")

for model_size in ${model_sizes[@]}; do
    for layer_type in ${layer_types[@]}; do
        echo $"------ Starting $model_size $layer_type \n"

        if (( $do_10s )); then
            python3 benchmark_layer.py --model-size $model_size --layer-type $layer_type --gpu-type $gpu_type --skip-gptq --skip-awq --use-10s-input
        else
            python3 benchmark_layer.py --model-size $model_size --layer-type $layer_type --gpu-type $gpu_type --skip-gptq --skip-awq
        fi
    done
done