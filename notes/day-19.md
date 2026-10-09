A file was a chunk only because I wrote it that way.

A chapter mixes topics. If I paste the whole chapter, the model sees the right fact and three distractors, and the window fills up.

If I cut too small, the sentence that answers the question is split from the sentence that defines the term.

Today I cut one chapter three ways and ask which cut puts the right passage on top.

Chunking is the process by which you create chunks from your corpus of knowledge. A chunk is the unit that you embed and that you paste. If your chunking decisions are not optimal, it can result in 3 common failure modes:

- Too wide: Your chunks are too big and contain a lot of irrelevant information which clogs up the model context unnecessarily.
- Too narrow: Your chunks are too small and no single chunk contains all of the information necessary to answer and your query might match will multiple chunks very loosely.
- Split facts: It can happen that you have one chunk where you define a term and a second chunks where the term is used for an answer but the question uses the term vaguely, so now you get the former chunk but the answer is in the latter chunk.

# Notes

How many chunks in each scheme:
- Whole: 1
- Fixed Window: 11
- Sections: 7

A chunk that was too wide: Whole file
A chunk that split a fact: Fixed window ("for giving the plant its green color. During photosynthesis, chlorophyll absorb")

Question where sections beat windows: almost all questions
Question where windows won or tied: Is chlorophyll responsible for a plant's green colour?

Read grade vs system grade, on the half-fact call: Section chunks won because of the biggest context length.

Cosine 0 means right angle, not opposite. (One sentence, so yesterday’s note does not stay wrong.) As mentioned yesterday, a cosine value of 0 means that the query text and the referene text are orthagonal i.e. unrelated, but not opposite meaning.

What I still find shaky, in full sentences:
- I'm still not sure we're doing the best possible use of our time with this course (for eg, we spent 1.5 days on multihead self attention transformers, but 3 days on embedding exercises). I would assume that the market values skills of someone who is able to write transformers and train networks from scratch (eg: provision a cloud GPU or let your code run overnight on your local computer, download this dataset from hugging face, write the code to train a model from scratch on this dataset to get the loss below some value) much higher than someone who can embed files for retrieval (which I'm guessing are common skills by even computer science seniors at this point). Let's please focus on skills that are in demand in the industry and try to compress exercises which are mostly gluing library code (like the embeddings we did this week)into as little time as possible (I reckon we could have done all embedding exercises in 1 day instead of 3 days if you had generated the corpus ahead of time and given it to me with the exact questions etc) Please think a little harder in the future.
- Whenever I'm running python code with `sentence-transformers` or `fastembed`, they seem to be downloading something from the internet each time. Is there a way to cache this so that the code is executed quickly (or I don't get banned for too many requests)?