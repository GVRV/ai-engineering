The hand score on Day 26 is the audit set. Twenty rows, seed 42, paste capped at 1,500 characters.

A judge model sees the paste and the reply. It does not see the SQuAD gold answer, and it does not see whether retrieval hit.

It answers one question: is every factual claim in the reply supported by the paste?

I compare its yes/no to my hand label. A disagreement is the result. Agreeing with myself is not.

Substring containment is a pre-filter. It cannot grade a paraphrase, and it cannot see an added clause. The judge can. The judge can also be wrong. That is why I read the disagreements instead of trusting the new column.

## Flips after normalization

Total: 1

- `How high is the highest point in Warsaw?` flipped because the answers ['452.8 ft', '115.7 metres', '115.7 metres'] contain decimals with points in them which get normalised to spaces in the paste context, so we can't find them.

- The question `Which countries are represented in the Far Eastern collections?` did not flip because the paste contains the normalised string "china japan korea" but the answers are ['China, Japan and Korea', 'China, Japan and Korea', 'East Asia: China, Japan and Korea'] so unless we remove stop words like "and" we run into the same issue.

Conclusion: This seems like one major con of this whole "find in paste" algorithm: the more we try to improve it, the more edge cases we run into. I highly doubt that anyone in the industry would be using this method for anything apart from structured data (inputs/outputs).

## Table with model judging

i | gold_hit | reply | answer in paste (normalised) | grounded | useful | judge
1 | true | 138 metres (452.8 ft) | false | true | true | YES; the paragraph states Szczęśliwice hill is 138 metres (452.8 ft).
2 | true | Matrix algebra (the use of matrices and elimination methods similar to Gaussian elimination) | true | false | true | YES; the paragraph states Zhu used rectangular arrays of coefficients equivalent to modern matrices.
3 | true | Pawiak | true | true | true | YES; the paragraph contains the word "Pawiak" and describes it.
4 | true | Contractors. | true | true | true | YES; reply contains no factual claim, so vacuously supported.
5 | true | a fee per unit of connection time, even when no data is transferred. | true | true | true | YES; the reply directly quotes the paragraph’s description of circuit switching fees.
6 | true | automobiles | true | true | true | YES; the paragraph contains the word automobiles.
7 | true | A progressive tax. | true | true | true | YES; the paragraph defines a progressive tax and the reply matches that definition.
8 | true | World War II | true | true | true | YES; the paragraph explicitly mentions World War II.
9 | true | He called it “my Patmos.” | true | true | true | YES; paragraph states Luther referred to Wartburg as “my Patmos.”
10 | true | The primes 41, 43, 47 and 53 appear in the **third étude** of *Neumes rythmiques*. | true | true | true | YES; the paragraph explicitly states those primes appear in the third étude, *Neumes rythmiques*.
11 | false | Great Yuan (大元) | false | false | false | YES; the paragraph explicitly states Kublai announced the dynasty as Great Yuan.
12 | false | NOT_IN_SOURCE | false | true | false | YES; paragraph does not contain the answer.
13 | false | Oxygen is used in cellular respiration. | true | true | true | YES; the paragraph explicitly states that oxygen is used in cellular respiration
14 | false | civil rebellion | true | true | false | YES; the phrase "civil rebellion" is explicitly mentioned in the paragraph.
15 | false | NOT_IN_SOURCE | false | true | false | YES; paragraph lacks answer.
16 | false | NOT_IN_SOURCE | false | true | false | YES; paragraph does not contain the answer.
17 | false | Carbon emissions. | false | false | false | YES; the paragraph discusses carbon release (emissions) from forest destruction.
18 | false | NOT_IN_SOURCE | false | false | false | NO; the paragraph provides specific plague death figures that answer likely questions.
19 | false | China, Japan, Korea | false | true | true | YES; The paragraph lists China, Japan, and Korea as part of the Asian departments.
20 | false | NOT_IN_SOURCE | true | false | false | NO; paragraph contains the answer.

# Notes

Answer model, judge model, temperature: openai/gpt-oss-20b, openai/gpt-oss-20b, 0
Did the judge see the gold span: no
Twenty reused, seed 42: yes

Net flips after normalization: 1
Judge agreement with hand grounded count: 17
Judge disagreements with hand grounded: 3

Three rows, who I think is right:
- For the question `What modern math concept did Zhu Shijie do work similar to?` the model judges the answer `Matrix algebra (the use of rectangular arrays of coefficients equivalent to modern matrices).` to be grounded, but I had judged it as not grounded because the model earlier included some references to "Gaussian elimination" in its reply but it's not doing so right now and we haven't saved the responses from earlier (because you didn't tell me responses should have saved/committed for future exercises).
- For the question `What is the Chinese name for the Yuan dynasty?`, the model judged the answer `Great Yuan (大元)` as grounded by the chinese symbols `大元` cannot be found in the text and the model used its knowledge to judge the chinese symbols. This seems like inference.
- For the question `A complete loss of rainforest cover may be caused by what type of emissions?` the model judged the answer `Carbon emissions.` as grounded, but the paste didn't contain the string "carbon emissions" even though there were references to "carbon contained within", "carbon stores", etc I think this was also some inference done by the model using its world knowledge.

Invalid grades: 0
Judge saw gold span: no
New answer-model calls: 0

What I still find shaky, in full sentences:
- I get that there are pros/cons of using the manual paste evaluation method (human judgement is hard, it is time consuming for big datasets, but it's nearly as good as we can get), there are also downsides to paste evaluation using just substring filtering (stop words, paraphrases get flagged, etc), so using another model as a judge is a good idea (but again models can hallucinate), so a better idea would be to have *multiple* models evaluate something and then get their consensus, right? Is this industry practice?