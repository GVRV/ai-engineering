Attention mixes information across time.

The feed-forward layer mixes information inside one token (across its channels).

A residual connection adds the input of a sub-layer back to its output, so the original signal is not overwritten.

LayerNorm rescales each token’s channels to mean about 0 and spread about 1, using only that token — no batch statistics, no running mean.

Those four pieces, in that order, are one Transformer block.

After fixing my code by adding a projection step within the MultiHeadAttension module, the loss is:
{'train': tensor(2.2676), 'val': tensor(2.2954)}

Then, with the addition of the feed forward MLP the loss slightly increases (I'm guessing because the bigger network takes more time to optimise) to:
{'train': tensor(2.2745), 'val': tensor(2.3016)}

After adding the projection operation to the FeedForward layer and increasing the size of the feed forward layer by 4 times (as mentioned in the original paper):
{'train': tensor(2.1804), 'val': tensor(2.2237)}

With the introduction of residual connections, the loss goes down to:
{'train': tensor(2.1142), 'val': tensor(2.1628)}

With the addition of LayerNorm, the loss goes down to:
{'train': tensor(2.1080), 'val': tensor(2.1661)}

Using sequential layer of 3 transformer blocks, the loss goes down to:
{'train': tensor(1.9866), 'val': tensor(2.0574)}

Adding a dropout (with probability of 0.2), the loss goes up slightly to:
{'train': tensor(2.0967), 'val': tensor(2.1417)}

Feed-forward does (and does not mix across time): It gives all of the values concatenated from the multiple heads ability to intermix and learn from each other other for a single token.

Residual means, in one sentence: Using the multiattention head and feedforward channel as a side-channel that gets added to the original input. This is so that the gradients can flow back to the input easily.

LayerNorm vs BatchNorm, in the table from Block D, filled with my words: In Batchnorm, we wanted to ensure that values coming across the batch (different inputs) were roughly gaussian so that we would not have any single input from a given batch that caused the network to learn in an inefficient manner. With LayerNorm the concept is the same, but we want the inputs of the multiattention head and feedforward layers to be roughly gaussian by themselves so that the values do not blow up or diminish over the calculations.

I used pre-norm (LayerNorm then sub-layer). Yes/no: Yes

Train loss: 2.1080
Val loss: 2.1661
Note: the loss went even lower once I added 3 transformer layers and a dropout (see above)
Yesterday four-head val: 2.29
Did this beat it: Yes

Parameter count (print sum(p.numel() for p in m.parameters())): 42369 (with 3 transformers and using dropout)

What I still find shaky, in full sentences:
- Why do we need a projection operation for each of the FeedForward/MultiHeadAttension modules? What's the purpose? I understand the purpose in the FeedForward layer considering the Transformers paper suggests using a hidden layer of 4x width, so using the projection layer we can get back to the number of embeddings width, but is that the purpose with the multiattention heads as well (in case the number of heads * head size doesn't equal the number of embeddings value)?