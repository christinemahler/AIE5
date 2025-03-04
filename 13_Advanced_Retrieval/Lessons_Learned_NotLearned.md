# 3 Lessons Learned

1. LangSmith tracing is temperamental. I spent a lot of time troubleshooting issues with getting the tracing to work even though I had everything configured correctly. The final solution I arrived at was needing to close and re-open cursor each time I started a new trace.
2. Not every retriever is created equal (I think that was a given for this session). 
3. There will almost certainly be some trade offs in the retriever that you choose, e.g. cost for accuracy, latency for precision, etc.

# 3 Lessons Not Learned

1. Are there alternatives to LangSmith?
2. What is the right balance of performance metrics? This still feels very "squishy."
3. What is the impact of the test data on each of the performance metrics? And how do you control for it? Would it be better to produce the test data in a different way?