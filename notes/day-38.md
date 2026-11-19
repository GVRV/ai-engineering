I built attention from scratch in week 3 and have not used the mechanism since.

Today I take one AG News sentence, run it to the first DistilBERT attention layer, and print the shapes of Q, K, and V.

Then I write what each matrix is. If I cannot, the blank is still there.

# Notes

Sentence: "Fears for T N pension after talks Unions representing workers at Turner   Newall say they are 'disappointed' after talks with stricken parent firm Federal Mogul."
seq length: 32 (i.e. tokenizer used 32 tokens for this sentence)
Q, K, V shapes: (1, 32, 768)
    - There's only one input sentence
    - That input sentence has 32 tokens
    - Each of those tokens gets embedded in a 768 dimensional space (because there are 12 heads of 64 size)
Heads, if you split: 12 heads of size 64 each

Four sentences (query, key, value, softmax direction):
    - The query part of the attention mechanism encodes what this particular token is looking for (eg: a verb might be looking for a subject)
    - The key part of the attention mechanism encodes what a particular token is (eg: a noun might want to signal that it can be a subject)
    - Once the query is mixed with the key, we perform a softmax on the direction of the key so that for each token, we calculate a distribution over the keys that most likely answer to its query
    - However, the keys are not feed back in. Instead the values are used in accordance with the distribution calculated. The keys signal "this box is relevant", the value is what's in the box as an analogy.
One token and what it attended to:
attention from 'fears':
  0.067  [CLS]
  0.026  fears
  0.035  for
  0.008  t
  0.021  n
  0.054  pension
  0.040  after
  0.041  talks
  0.036  unions
  0.036  representing
  0.028  workers
  0.027  at
  0.029  turner
  0.013  new
  0.018  ##all
  0.037  say
  0.032  they
  0.046  are
  0.036  '
  0.043  disappointed
  0.035  '
  0.036  after
  0.037  talks
  0.021  with
  0.033  stricken
  0.024  parent
  0.037  firm
  0.019  federal
  0.007  mo
  0.011  ##gul
  0.025  .
  0.043  [SEP]

What q_lin adaptation changes: This changes what each token is looking for (i.e. during LoRA we can change the query to behave from "hey, i'm a verb looking for subject" to "hey, I'm a verb looking for an adverb")
What v_lin adaptation changes: This changes what gets mixed into the result for the feed forward layers (i.e. considering you opened this box, now what gets mixed is different than before so what will feed forward might be different values).
The key is not changed because we want to keep the signaling of the tokens the same as during pre-training (signaling that this is a "noun" box that might be a good subject, etc) because considering the Queries and Values were modified, the behaviour of the network changes without any changes to the "this is what I am" advertising that is performed by the key mechanism.

Opened the week-3 code to write this: yes

What I still find shaky, in full sentences:
- How do people determine how many attention heads to put into their models and the head size? Do the head size need to stay the same between attention heads? Why/why not?
