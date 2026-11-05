With one tool, every call is a retrieve. I cannot tell a good agent from a model that always calls the only thing on offer.

The second tool is the AG News classifier, GVRV/ag_news. It labels a headline World, Sports, Business, or Sci/Tech.

A question about a fact in the Wikipedia paragraphs should call retrieve. A question that asks what category a headline is should call classify.

Calling the wrong one is the failure this day exists to count.

# Notes

Tools: retrieve, classify
Correct routes: 10/10
Wrong routes, with the question and the tool: 0/10
Classifier label on the Intel headline: "Sci/Tech"
Did the factual rainforest question hit retrieve: Yes, 1 "retrieve" call

What I still find shaky, in full sentences:
- I appreciate that you're trying to get me to use my own lego blocks but considering we didn't write them as lego blocks, your insistance on using previously written code in new ways means that most of the day is spent on refactoring the code to not break in the old way of working and still be useful for the new day instead of learning the concept. It would be helpful if we can avoid long chains of building on previously written code or alternatively, you can instruct me to design components such that they are easily used for future projects from the get go (eg: if our classifier had to work as a tool, you could've given me signatures for functions such that the tool already worked but I didn't know about it).