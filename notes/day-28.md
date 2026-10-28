A string check is not a judge of grounding. It answers one question: is the gold answer text inside the paste the model saw?

If it is, the row cannot be an invention of that fact. It may still be an inference, like Far Eastern.

If it is not, and the model stated it anyway, that is the invention bucket.

The check does not replace the hand score. It catches the class of mistake we made.

# Three interesting rows

### First - Rudimentary `paste` logic doesn't help

For the question `Squad Question: Which countries are represented in the Far Eastern collections?`, the paste context has the string "China, Japan, Korea" but our answers are "China, Japan and Korea" which means our rudimentary `paste` logic thinks the answer is not in paste when it kind of is. This feels like we're using the kindergarten version of the `paste` logic instead of what even mildly competent people in the industry might be using.

### Invention

For the question `A complete loss of rainforest cover may be caused by what type of emissions?`, the golden answers ['greenhouse gas emissions', 'greenhouse gas', 'greenhouse gas'] are not in the paste context, but the model still replied with `Carbon emissions`.

For the question `What is the Chinese name for the Yuan dynasty?` the golden answers ['Yuán Cháo', 'Yuán Cháo', '元朝'] are not in the paste but the model still responded with an invention "Great Yuan (大元)".

# Notes
Seed, n: 42, 1500 characters paste length
answer_in_paste true count: 12
False refusals (in paste, model said NOT_IN_SOURCE): 0
Inventions the check agrees with (not in paste, but replied correctly): 2
Rows the check would have saved me from mislabeling: 1-2 maybe (again, because the logic for `paste` is so rudimentary, we can't rely on it as well)

Did this replace the hand score: no
New API calls: no

What I still find shaky, in full sentences:
- I doubt anyone in the industry is using substring look-ups to get an idea if the golden answer(s) were present in the paste context. Today felt like a waste of time.