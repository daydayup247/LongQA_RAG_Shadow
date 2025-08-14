# A Collaborative Reasoning Framework for Large Language Models in Long-context Q&A

This repository implements the Collaborative Reasoning Framework for improving the performance of LLMs on long-context Q\&A.

## Abstract

Large Language Models (LLMs) often struggle with the Lost in the Middle phenomenon in long-context question answering (Q\&A). Existing solutions, such as modifying attention mechanisms or positional encodings, typically require retraining, which demands substantial computational resources. Other strategies, including long-term memory mechanisms and context processing, heavily rely on auxiliary components and fail to fundamentally enhance the LLM's reasoning capabilities. To bridge this gap, we propose a novel collaborative reasoning framework. Initially, the framework uses a retrieval-augmented generation (RAG) approach to generate a candidate answer from sentences relevant to the input question. Subsequently, a training-free Shadow-LLM is designed to supplement local sentence-level information from the long context during the reasoning process to produce another candidate answer. Finally, a one-out-of-two selection strategy chooses the final answer based on the two candidates. Experiments on three long-context Q\&A datasets show that our method raises the F1 score over the baselines by 2% to 18%. Notably, we find that activating only the $0$-th decoder layer of the LLM is sufficient for Shadow-LLM to operate at optimal performance, enabling efficient deployment without retraining.

## Code Workflow

The code in this repository follows a four-step workflow:

1. **RAG Indexing**  
   - **File:** `./rag/index_building_bm25.sh`
   - **File:** `./rag/index_building_dense.sh`  

2. **LongQA-RAG Inference**  
   - **File:** `pred_llama2_rag_sparse.py`
   - **run:** `python pred_llama2_rag_sparse.py --model llama2-7b-chat-4k-rag`
   
   - **File:** `pred_llama2_rag_dense.py`
   - **run:** `python pred_llama2_rag_dense.py --model llama2-7b-chat-4k-rag`

3. **Shadow-LLM Inference**  
   - **File:** `pred_llama2_shadow.py`
   - **run:** `python pred_llama2_shadow.py --model llama2-7b-shadow`
  
4. **Answer Refinement**  
   - **File:** `discriminator_llama2.py --model llama2-7b-chat-4k`
   - **run** `python discriminator_llama2.py --model llama2-7b-chat-4k`
