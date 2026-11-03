Yesterday the model called `retrieve` once. The query was “Levi's Stadium opening year.” No paste had been returned yet. The model already knew the stadium.

Today the system prompt forbids naming a stadium, a year, or a team unless a paragraph the tool returned contains it.

I am measuring whether that ban produces a first query that does not contain “Levi's.”

If the first query still says Levi's, the prompt did not constrain the model. That is a result. I will not tune the prompt until it complies.

# Notes

Prompt change, in one sentence: We modified the prompt to ban the system from using its world knowledge for retrieval queries from the start; and instead start building knowledge with just the question first.
Question: In what year did the stadium that hosted Super Bowl 50 open?
First query: "Super Bowl 50 stadium opening year"
Contains Levi / year / team before any paste: no
Outcome: hop didn't happen but the ban was not ignored either. It retrieved and found the answer in just one hope.
Calls: 1

Second question run: yes
If yes, first query: "Denver Broncos owner company founded by owner"
Second query: "Denver Broncos owner founded company"
Third query: "John Elway company founded"
Outcome: STOPPED
Calls: 3
Note: The model used its world knowledge from the first query (how did it know Denver Broncos won Super Bowl 50?)

Did I rerun to improve the log: no
Did I change the tool or the cap: no

What I still find shaky, in full sentences:
- I understand the point that we can get the model to behave better by optimising the system prompt, but we're not using SOTA models so they might have edge cases. I didn't learn much today because all we did was optimise the system prompt and it still didn't produce a hop for our original question, and neither did it work better on the question we invented.