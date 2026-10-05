Yesterday’s tokeniser merges the most common byte pair anywhere, including across a space.

GPT-2 first cuts the string into chunks (letters, numbers, punctuation) and only merges inside a chunk.

Special tokens like an end-of-text marker are not merges. They are extra ids that never get produced by the merge loop.

The tokeniser is trained on its own data, before the Transformer ever runs.

### Block A

- Why 123 should not be allowed to merge with the letter after it.

Because merging numbers and letters this way would lead to errors where the `123` context is needed for arithmetic responses. Merging of numbers and letters this way leads to tokens which cannot be correctly used in responses that require numbers to be parsed and processed by themselves.

- What a special token is, and why it is not in the merge list.

A special token is a token that is usually introduced during the fine-tuning stage of the model to signify meta-data in how we want to train the model to answer as a chat assistant. An example is the <|endoftext|> token used by GPT-2 to signify the end of a certain document used to wipe its "memory" clean for the content that might follow.

- One “LLM quirk” that is actually a tokeniser quirk (trailing space, spelling, the word “strawberry”).

A lot of ways you can shoot yourself in the foot with tokenization. Trailing space is one because for certain models, the tokens are merged with a "starting space" (eg: " world") so if there is a trailing space at the end of the text, the model finds it difficult to predict the next token. Spelling is another issue because sub-strings in a word might be merged into 1 or more tokens for efficient parsing, but this leads to the model not knowing the exact composition of the token in letters which can be used for answering spelling questions.

### Block B

First 10 Merges:
--> t<--:404
-->he<--:383
-->ou<--:270
--> a<--:255
--> the<--:196
-->re<--:194
--> w<--:194
--> s<--:185
-->in<--:183
-->ha<--:164

Starting length of text: 20000
Old text length: 20000
New text length: 9301
Compression: 2.15
Length of decoded input: 20000

### Block E

What I built:
  character Transformer: n_embd 384, 6 blocks, context 256
  train 1.06, val ~1.48 then 1.49
  weights not saved
  BPE: 300 merges, 20000 -> 8823, round-trip yes
  forced splits: first merges differ because now we're only looking at the merges occuring within splits and not across splits.

Tokenizer quirks I can now name: Spelling errors, Arithmetic issues, unactivated vocabulary, Special tokens

Pretrain / fine-tune / chat, in one sentence each:
- During the pre-training stage, we're training the model over a large dataset ("entire internet") to become an unaligned document completer model (i.e. predict the next set of characters/words)
- After training, we fine-tune the model on documents that have a question/answer structure so that the model gets aligned to expecting a question and then generating an answer.
- After fine-tuning, there are can many other steps like Reinforment Learning from Human Feedback to create a chat model that is actually helpful and effective.

What I still find shaky, in full sentences: A lot of the papers seems quite recent (less than 10 years ago) and a lot of the code seems work-in-progress (eg: we were able to generate a tokenizer used by GPT-2 in an afternoon), so I'm guessing the field is moving at a blinding pace and there are a lot of small improvements/optimizations happening that we're skipping over (focusing on breadth right now instead of depth)?

