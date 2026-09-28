Last two weeks we predicted the next letter from a fixed window of previous letters, using a table or a small MLP.

A Transformer still predicts the next letter (or token), but each position chooses how much to read from each earlier position.

Today we build that “choose” as: make a weight for every pair (this position, earlier position), make those weights add to 1, multiply them by the earlier values, sum.

We will not yet let the model learn those weights. That is query/key, tomorrow.

How this dataset differs from names.txt: It's much bigger (1 million chars vs ~300K), and has a bigger vocabulary (~65 vs ~27). We can also assume that while the previous dataset only resulted in one word predictions for a single name, here we want to generate arbitrary text so there might be deeper structure between text that would need to be trained on.

What B and T mean in a batch: B is the batch number, each batch can be trained on independently (I'm also assuming each batch can be inferenced from independently). T is the time component, which gives us the ability to look at block_size of predictions that were made in the past to better predict the future token.

Bigram baseline:
  step-0 loss (should be near 4.17): 4.3695
  loss after a short train: 2.5068
  a few sample lines:

```
is t sormmy wit wis timariangs.
RKINelags,
LR:
f ojefer ald;
Y s thelf VORerelelil hak
Wheseveer:
G
```

Version 1 (loops) does: iterate over each batch, iterate over each time component in that batch, sum up the channels for the time components from 0 until the current element, calculate the mean and then set it to the current time component.

Version 2 (matrix multiply) is the same because: We're just using more efficient math to calculate essentially the same values. Using matrix multiplications, we can parallelise the math efficiently but get the same results.

Why we put -inf above the diagonal: If we're assuming that weights in the softmax version will be used to signal any affinity that previous predicitons might have on the future prediction, then -inf will signify that there is zero coupling between the letters. Essentially, this means that future predictions cannot affect the predictions we're making right now.
So a character cannot look at characters that have not been written yet.
That is “causal” attention: cause stays in the past.

Softmax on a row does: normalise the row to probablities summing up to 1.0

I did not implement query/key/value today.

What I still find shaky, in full sentences:
- We didn't add <END> and <START> tokens to the token vocabulary? I can understand that <START> is not needed if the input starts at T = 0 but where do we end?
- Andrej mentioned other tokenizing techniques like SentencePiece and Tiktokenizer and algorithms like Byte Positioning Embedding? Do we need to learn these (at least theoretically for our rounded education)?
- We're getting more deeper in the PyTorch API with nn.Module, Embedding, AdamW optimizer, etc. I get these are lego blocks and swappable but do I need to understand pytorch api design and theory before using these?
- For now, I'm just assuming that in the future, we will find a better way to use past probabilities to affect future predictions because averaging out the parameters feels like it's not a great idea.