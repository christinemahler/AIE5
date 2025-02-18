# AIE5 Assignment 9 - Fine Tuning Embedding Models

GitHub Link: https://github.com/christinemahler/AIE5/blob/main/09_Finetuning_Embeddings/Fine_tuning_Embedding_Models_for_RAG_Solution_Notebook%20COMPLETED.ipynb

Loom Video: https://www.loom.com/share/00b6ad0426b245a8aeaa624f582d77b2?sid=48a8a279-020e-4096-bdd3-24fdca0ddd67

# 3 Lessons Learned

1. Fine-tuning the embedding model can significantly improve the performance of the LLM application especially when working with domain-specific use cases.
2. Fine-tuning is an iterative process.
3. RAGAS evaluation is an essential tool for evaluating the results of our fine-tuning process.

# 3 Lessons Not Learned

1. Why we do not use the RAGAS framework to generate training, test, and validation datasets.
2. What is the most appropriate batch size, warmup steps, epochs, evalution steps, etc. Recommended practice for fine-tuning configurations.
3. Does chunking strategy have an impact on fine-tuning process? Can this result in poorly generated training, test, and validation datasets?