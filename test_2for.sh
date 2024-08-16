#!/bin/bash

datasets=("knights" "bracelet" "bunny" "tarot")
batch_sizes=("2048" "4092" "16384")

for dataset in "${batch_sizes[@]}"; do
    for batch_size in "${datasets[@]}"; do
        echo "Dataset: $dataset, Batch size: $batch_size"
    done
done
