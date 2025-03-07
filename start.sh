#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
export NCCL_ALGO=Tree
export PYTHONUNBUFFERED=TRUE

gunicorn -w 2 --worker-class=gthread --threads 4 \
--preload \
--timeout 300 \
--bind 0.0.0.0:8888 \
--access-logfile - \
--error-logfile - \
"src.api.llama_handler:app"