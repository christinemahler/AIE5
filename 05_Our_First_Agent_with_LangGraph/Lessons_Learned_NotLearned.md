# 3 Lessons Learned

1. Agent patterns simply go from acyclic to cycle behavior in our LangGraph graphs, but there's an inherent risk to creating an infinite loop. This can be controlled either by counting the total number of messages added to the state object (as we do in our example code this week) or else by adding a recursion_limit value to our state object and handling the resultant error in a conditional edge.
2. Agents determine which tools to call implicitly based on a combination of the user prompts and tool docstrings, but this behavior can also be overriden by pass in a [tool choice](https://python.langchain.com/docs/how_to/tool_choice/) parameter.
3. LangSmith evaluators are helpful for determining the validity of a model's output, but they require the creation of reference datasets.

# 3 Lessons Not Learned

1. How to use LangSmith for unit testing.
2. How exactly a tool is selected by the LLM for use.
3. How better to avoid the infinite loop problem than simply setting a recursion limit. It seems like we could check the last response against the current response to close out the loop before it ever prints a response to the screen.