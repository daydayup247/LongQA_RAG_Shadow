#!/bin/bash

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input rag_data_bm25 \
  --index indexes/qasper \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw