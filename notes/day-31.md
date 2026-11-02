`loop.py` always retrieves once, then answers. The model never chooses.

An agent gets a tool. It may call it, call it again with a different query, or stop.

I execute the tool. The model does not run code. It returns a name and arguments. I run the function and send the result back.

A YES from the judge still only means “supported by the pastes the model actually saw.”

# Questions

Question: "In what year did the stadium that hosted Super Bowl 50 open?"
Calls: 1
Queries: "Levi's Stadium opening year"
Reply: 2014
Judge: yes
Facts in the pastes: stadium yes, year yes
Notes: The model used its world knowledge to determine immediately that the stadium was Levi's stadium and then used tool call to find text supporting when it was built.

Question: "What is the name of the company founded by the owner of the team which won Superbowl 50?"
Calls: 3
Queries:
    - "John Elway company founded"
    - "Super Bowl 50 winner team owner founded company"
    - "John Elway company founded"
Reply: Error (it continued to tool call even after the option was not passed in resulting in an error from Groq)
Notes: I asked it a "out of syllabus" question deliberately to understand if multiple tool calling is working as expected.

# Notes

What I still find shaky, in full sentences:
- I understand tool calling but it seems like flakey as hell (especially with older/smaller models which continue to tool call even after it is no longer an option, or sometimes it tool calls just once and determines there is no answer in the source instead). There needs to be better management of tool calling (or at least model awareness) to use the optimum number of calls before making an informed decision.

