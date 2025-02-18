# 3 Lessons Learned

1. Fine-tuning the embedding model can significantly improve the performance of the LLM application especially when working with domain-specific use cases.
2. Fine-tuning is an iterative process.
3. RAGAS evaluation is an essential tool for evaluating the results of our fine-tuning process.

# 3 Lessons Not Learned

1. Why we do not use the RAGAS framework to generate training, test, and validation datasets.
2. What is the most appropriate batch size, warmup steps, epochs, evalution steps, etc. Recommended practice for fine-tuning configurations.
3. Does chunking strategy have an impact on fine-tuning process? Can this result in poorly generated training, test, and validation datasets?