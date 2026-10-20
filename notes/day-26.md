Yesterday hit@1 was 0.625. On about a third of questions the top paragraph is the wrong page.

Today I paste that top paragraph into the prompt and ask the model to answer from it.

I score the reply against the pasted paragraph, not against the SQuAD gold context.

If the paste does not contain the fact, the only grounded reply is a refusal. A fluent right answer from the model's own memory is an invention relative to the paste.

# Notes

Model id, temperature: openai/gpt-oss-20b, 0
How many questions, seed: 20, 42
Hit@1 from yesterday, for reference: 0.625

Table of 20:
i | gold_hit | reply | grounded | useful | one line
1 | true | 138 metres (452.8 ft) | true | true |
2 | true | Matrix algebra (the use of matrices and elimination methods similar to Gaussian elimination) | true | true | Matrix algebra is grounded, but Gaussian eliminiation is not
3 | true | Pawiak | true | true |
4 | true | Contractors. | true | true |
5 | true | a fee per unit of connection time, even when no data is transferred. | true | true |
6 | true | automobiles | true | true |
7 | true | A progressive tax. | true | true |
8 | true | World War II | true | true |
9 | true | He called it “my Patmos.” | true | true
10 | true | The primes 41, 43, 47 and 53 appear in the **third étude** of *Neumes rythmiques*. | true | true |
11 | false | Great Yuan (大元) | false | false | Context contains the text "Kublai announced the name of the new dynasty as Great Yuan" but the mandarin characters aren't in the context
12 | false | NOT_IN_SOURCE | true | false |
13 | false | Oxygen is used in cellular respiration. | true | true |
14 | false | civil rebellion | true | false | The model confuses civil rebellion in the corpus with civil disobedience
15 | false | NOT_IN_SOURCE | true | false |
16 | false | NOT_IN_SOURCE | true | false |
17 | false | Carbon emissions. | false | false | The corpus contained a lot of references to carbon stored in rainforests but didn't mention carbon emissions at all
18 | false | NOT_IN_SOURCE | false | false | The source starts with "In 1466, perhaps 40,000 people died of the plague in Paris" so a much more useful, grounded reply would be at least 40,000.
19 | false | China, Japan, Korea | false | true |
20 | false | NOT_IN_SOURCE | true | false |

Bucket counts:
- gold hit, grounded, useful: 10
- gold hit, not useful: 0
- gold miss, refused or answered from paste: 6
- gold miss, answered with fact not in paste: 4

One row where the model knew the answer and the paste did not: "Which countries are represented in the Far Eastern collections?"
One row where the paste had it and the model refused: "How much of Paris' population was killed by the plague?"

Did I score against the paste, not the gold page: yes
Did I call the API on all 10570: no

What I still find shaky, in full sentences:
- Nothing really, I guess a more verbose system prompt to truly isolate the paragraph as the only available source for the answer might result in somewhat better grounding.