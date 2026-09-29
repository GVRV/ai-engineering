Yesterday each position built a new vector by averaging every earlier vector equally.

Today each position emits three vectors from its embedding:

a query (what I am looking for),

a key (what I contain, so others can match against me),

a value (what I hand over if someone attends to me).

The weight between “me now” and “you earlier” is how well my query matches your key.

The new vector at my position is a weighted sum of values.

Future positions still get no weight.

Some debugging checks:
- Tril doesn't show up in m.parameters() - confirmed with print statement
- Weights have -INF and rows add up to 1:
```
> /Users/gaurav/Projects/ai-engineering/experiments/gpt/one_head.py(85)forward()
-> wei = q @ k.transpose(-2, -1) * C**-0.5 # B, T, C @ B, C, T -> B, T, T
(Pdb) n
> /Users/gaurav/Projects/ai-engineering/experiments/gpt/one_head.py(87)forward()
-> wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
(Pdb) wei[0]
tensor([[-0.6295, -0.4504,  0.8334,  0.3931,  0.0510,  0.2338, -0.5878, -0.5966],
        [-1.1496,  0.0366, -0.0918, -0.4823, -0.2185, -0.8302, -0.0857,  0.6123],
        [-0.0422, -0.5914, -0.2470,  0.0239, -0.2126, -0.2156, -0.0607, -0.0150],
        [-0.1774, -1.4427, -0.5257,  0.0133, -0.7048,  0.0583,  0.4285, -0.4391],
        [ 0.6666, -0.9092, -1.0731, -0.8196, -0.2881, -0.0495,  0.3725, -0.6861],
        [ 1.1753, -0.5234, -1.3950, -0.2205, -0.0525, -0.1162,  0.8272, -0.6826],
        [ 0.4909,  0.1355, -0.3995,  0.5700,  0.0158,  0.8036, -0.4200,  0.2400],
        [-1.0003, -0.1978,  0.5219,  0.1357, -0.2400, -0.5235, -0.0097,  0.2675]],
       grad_fn=<SelectBackward0>)
(Pdb) n
> /Users/gaurav/Projects/ai-engineering/experiments/gpt/one_head.py(88)forward()
-> wei = F.softmax(wei, dim=-1) # B, T, T
(Pdb) wei[0]
tensor([[-0.6295,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
        [-1.1496,  0.0366,    -inf,    -inf,    -inf,    -inf,    -inf,    -inf],
        [-0.0422, -0.5914, -0.2470,    -inf,    -inf,    -inf,    -inf,    -inf],
        [-0.1774, -1.4427, -0.5257,  0.0133,    -inf,    -inf,    -inf,    -inf],
        [ 0.6666, -0.9092, -1.0731, -0.8196, -0.2881,    -inf,    -inf,    -inf],
        [ 1.1753, -0.5234, -1.3950, -0.2205, -0.0525, -0.1162,    -inf,    -inf],
        [ 0.4909,  0.1355, -0.3995,  0.5700,  0.0158,  0.8036, -0.4200,    -inf],
        [-1.0003, -0.1978,  0.5219,  0.1357, -0.2400, -0.5235, -0.0097,  0.2675]],
       grad_fn=<SelectBackward0>)
(Pdb) n
> /Users/gaurav/Projects/ai-engineering/experiments/gpt/one_head.py(90)forward()
-> v = self.value(x) # B, T, C
(Pdb) wei[0]
tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.2339, 0.7661, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.4180, 0.2414, 0.3406, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.3127, 0.0882, 0.2207, 0.3784, 0.0000, 0.0000, 0.0000, 0.0000],
        [0.5016, 0.1038, 0.0881, 0.1135, 0.1931, 0.0000, 0.0000, 0.0000],
        [0.4820, 0.0882, 0.0369, 0.1194, 0.1412, 0.1325, 0.0000, 0.0000],
        [0.1791, 0.1255, 0.0735, 0.1938, 0.1113, 0.2448, 0.0720, 0.0000],
        [0.0478, 0.1066, 0.2190, 0.1488, 0.1022, 0.0770, 0.1287, 0.1698]],
       grad_fn=<SelectBackward0>)
```

Query is: A mechanism of communicating what a token is looking for (eg: a vowel might be looking for consonants in the position before it)
Key is: A mechanism of communicating what a token represents (eg: a vowel might want to indicate that it should be high affinity if it was preceeded by certain consonants)
Value is: A mechanism linearly transform inputs so that we can get a better prediction using the affinities calculated from query and key
The score between position t and position j is: the affinity between the tokens i.e. how much they care about each other
Why we divide by sqrt(head size), in ordinary language: As we multiply the query and key, we get a variance that is roughly the head size and considering this score/weights will be softmaxed later on, we particularly don't have large values to dominate the probabilities. To avoid this and bring the variance of the weights/score back to around 1, we divide by the square root of the head size
Why the future is still -inf: In this particular model, we want to ensure that future tokens cannot communicate their affinities with the past token (because in this model we want to predict future tokens and if we allow this communication, then our parameters will just tune for these affinities). So, we add decoder block with a -inf value to signify a 0 probability affinity between future tokens and past tokens. In some other cases, like sentiment analysis, we actually do allow future tokens to communicate with past tokens and we don't add a decoder block.

Checks:
  weights row sums to 1: yes
  future weights are 0: yes

Train loss: 2.3960
Val loss: 2.4215
Yesterday's val (token+position baseline): ~2.51
Did this beat it: yes.

What "self-attention" means here:
Each position attends to other positions in the same sequence,
including itself. Nobody from another batch item is visible.

What I still find shaky, in full sentences:
- Andrej sort of said what q/k/v stand for, but I'm guessing there is some math behind why this mechanism leads to better predictions (why use Linear layers, how head size is determined, what if we only use q/v and not k or k/v but not q, etc)?
- When we started with Version 4/simple one-head attention exercise, the output changes shape from [4, 8, 32] to [4, 8, 16] which means that we cannot predict using the channels, right? This was resolved in the final example where we added the single head to the model by using NUM_EMBEDDINGS as the head size.