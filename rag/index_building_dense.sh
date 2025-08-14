#!/bin/bash

CUDA_VISIBLE_DEVICES=4,5 python -m pyserini.encode \
  input   --corpus rag_data_dense/qasper.jsonl \
  output  --embeddings indexes_dense/qasper \
          --to-faiss \
  encoder --encoder /home/jcyao/longtext/tct_colbert-v2-hnp-msmarco \
          --batch 32 \
          --fp16  # if inference with autocast()