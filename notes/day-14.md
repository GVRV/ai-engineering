### Yesterday feedback

Good tip about `model.save()` in PyTorch. I didn't know about it (but I did wonder how they distribute the model training across GPUs/computers and save progress). Was this supposed to be covered in one of our previous lectures? I will attempt to study `nanoGPT` codebase from Andrej to better understand checkpoints and training distribution/resuming.

### Notes

Yesterday the model could see 256 characters. That is a short paragraph, not a scene.

A tokeniser replaces common character sequences with a single integer, so 256 tokens can cover a page.

Byte-pair encoding starts from raw bytes and repeatedly merges the pair that appears most often.

The merges are a list. Encoding is “apply those merges.” It is not a neural net.

### Block B
Using `cl100k_base` encoding, in a single "To be..." sentence:
Number of characters: 42
Number of tokens: 13
Compression: 3.23

Using the first 2000 characters from tiny shakespeare:
Number of characters: 2000
Number of tokens: 497
Compression: 4.02

### Answers

Why 256 characters was a short context: Because we were using character encoding, the context just meant the transformer only had a context of 256 characters which is very small (around ~40 words). This means the output is determined from a very small sample of input information and using more efficient encoding of tokens, we can increase the output quality (eg: can you get more context about a given problem in 40 words or 100 words?)

Character vs word vs byte-pair, in one sentence each: Character encoding is 1 character is one token, word encoding is one word is one token, and byte-pair encoding a middle-ground solution where we increase the vocabulary by creating a new token for commonly occuring byte pairs.

My first 10 merges, written as characters:
-->e <--:517
-->th<--:402
-->t <--:321
-->s <--:291
-->ou<--:270
-->, <--:248
-->d <--:234
-->r <--:203
-->in<--:183
-->an<--:170

Round-trip worked: yes
Length before merges: 20000
Length after 50: 13665
Length after 300: 8823
Characters per token: 2.27

What I would have to change in transformer_block.py to train on these ids
(I did not do it): Using the new size of the vocabulary, I would have to change the embedding table to this size so that each of the new tokens has a row of vectors which it corresponds to. The sampling would also change as we're not just spitting out tokens anymore, we would need to decode the output to lookup byte pairs encoded by our new vocabulary.

Yesterday's scaled run, so I do not lose it:
  n_embd 384, 6 blocks, 6 heads, context 256, 5000 steps
  train 1.06, val bottomed ~1.48 then rose to 1.49
  sample looked like verse
  weights were not saved

What I still find shaky, in full sentences: Surprisingly, I don't find many things shaky about tokenization so far (I even did the decode code by myself without watching Andrej, so it might be suboptimal but it definitely works!). I understand that there are tradeoffs to extremely large vocabulary size that might encode information very densely and something like character-level encoding which is a very small vocabulary but packed very inefficiently. I'm guessing there is some complex math to determine the sweet spot.