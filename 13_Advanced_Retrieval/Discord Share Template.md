# AIE5 Assignment 13 - Advanced Retrieval

GitHub Link: https://github.com/christinemahler/AIE5/blob/main/13_Advanced_Retrieval/Advanced_Retrieval_with_LangChain_Assignment.ipynb

Loom Video: https://www.loom.com/share/2e00f220b15944c1a623826e7391d5cd?sid=c5e0d1bb-2eeb-447a-ac58-304a63818924

# 3 Lessons Learned

1. LangSmith tracing is temperamental. I spent a lot of time troubleshooting issues with getting the tracing to work even though I had everything configured correctly. The final solution I arrived at was needing to close and re-open cursor each time I started a new trace.
2. Not every retriever is created equal (I think that was a given for this session). 
3. There will almost certainly be some trade offs in the retriever that you choose, e.g. cost for accuracy, latency for precision, etc.

# 3 Lessons Not Learned

1. Are there alternatives to LangSmith?
2. What is the right balance of performance metrics? This still feels very "squishy."
3. What is the impact of the test data on each of the performance metrics? And how do you control for it? Would it be better to produce the test data in a different way?