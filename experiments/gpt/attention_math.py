#
# Attention, at this stage, is only: each position builds a new vector
# by adding up earlier vectors with some weights.
#
# Today the weights are “equal average of the past.”
# In the future, the weights will depend on the content.
#
# We can think of this as a prediction sequence which has reached the following:
#  0 | 1 | 2 | 3 | 4 | 5 <-- T (time)
#  h | e | l | l | o | ?
#
# When we're at index 3 and we know it's an `l` and trying to predict the letter
# at index 4, instead of just looking at the logits for what follows `l`, we
# can take the past into our context. i.e. we know that our current `l` was preceeded
# by another `l`. We also know that both those `l`s were preceeded by a `e` etc.
# We are able to look into the past until our block_size/context_window
# and using this, we might be able to make better predictions.
#
# Forget that just averaging the previous logits might not be the best way
# to share information across tokens (we will improve the math in the future)
#

import torch
from torch.nn import functional as F
from torch import nn
torch.manual_seed(1337)

B, T, C = 4, 8, 2 # batch, time, channels
x = torch.randn(B,T,C)
# print(x)

#
# For each single batch element
# For each time element `t`
# We want to average out the channels over the previous `t` elements
# up until the `t`th element
#

# Version 1: manual looping - works but inefficient
xbow = torch.zeros((B,T,C)) # bow = bag of words (convention for averaging out things)
for b in range(B):
    for t in range(T):
        xprev = x[b,:t+1] # (t, C)
        xbow[b, t] = torch.mean(xprev, 0)

# Version 2: matrix multiplication
a = torch.ones(3, 3)
# tensor([[1., 1., 1.],
#         [1., 1., 1.],
#         [1., 1., 1.]])
a = torch.tril(a) # Easy way to keep the lower half of the triangular values
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
a = a / a.sum(1, keepdim=True) # Make the rows add up to 1 so matrix multiply will average
# tensor([[1.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000],
#         [0.3333, 0.3333, 0.3333]])
b = torch.randint(0, 10, (3, 2)).float()
# tensor([[8., 6.],
#         [5., 2.],
#         [4., 4.]])
c = a @ b
# tensor([[8.0000, 6.0000],
#         [6.5000, 4.0000],
#         [5.6667, 4.0000]])
# print(a)
# print(b)
# print(c)

# Now with x, use weights that will average out using matrix multiply
wei = torch.ones(T, T) # T, T
wei = torch.tril(wei) # T, T
wei = wei / wei.sum(1, keepdim=True) # Make the rows add up to 1
xbow2 = wei @ x # Broadcasting: B, T, T * B, T, C
print(torch.allclose(xbow, xbow2))

# Version 3: using softmax - this is more interesting because the weights start with 0
# and we might be able to change this value in the future to indicate how much coupling
# there is between letters (affinity). Therefore a -inf affinity means they cannot be
# coupled at all and we're using that to indicate the future cannot communicate with the
# past.
wei = torch.zeros(T, T)
# tensor([[0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])
tril = torch.tril(torch.ones(T, T))
# tensor([[1., 0., 0., 0., 0., 0., 0., 0.],
#         [1., 1., 0., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 0., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 0., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 0., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 0., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 0.],
#         [1., 1., 1., 1., 1., 1., 1., 1.]])
wei = wei.masked_fill(tril == 0, float('-inf')) # Future cannot communicate with the past
# tensor([[0., -inf, -inf, -inf, -inf, -inf, -inf, -inf],
#         [0., 0., -inf, -inf, -inf, -inf, -inf, -inf],
#         [0., 0., 0., -inf, -inf, -inf, -inf, -inf],
#         [0., 0., 0., 0., -inf, -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., -inf, -inf, -inf],
#         [0., 0., 0., 0., 0., 0., -inf, -inf],
#         [0., 0., 0., 0., 0., 0., 0., -inf],
#         [0., 0., 0., 0., 0., 0., 0., 0.]])
wei = F.softmax(wei, dim=-1) # Why not use dim=1?
# tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.5000, 0.5000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.3333, 0.3333, 0.3333, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2500, 0.2500, 0.2500, 0.2500, 0.0000, 0.0000, 0.0000, 0.0000],
#         [0.2000, 0.2000, 0.2000, 0.2000, 0.2000, 0.0000, 0.0000, 0.0000],
#         [0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.1667, 0.0000, 0.0000],
#         [0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.1429, 0.0000],
#         [0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250, 0.1250]])
xbow3 = wei @ x
print(torch.allclose(xbow, xbow3))

# Version 4: self-attention!
torch.manual_seed(1337)
B,T,C = 4, 8, 32 # batch, time, channels
x = torch.randn(B,T,C)

# a single Head perform self-attention
head_size = 16 # hyperparameter
# Key is used to inform other tokens this is what I contain
key = nn.Linear(C, head_size, bias=False) # C, H
# Query is used. to inform other tokens what I'm looking for
query = nn.Linear(C, head_size, bias=False) # C, H

k = key(x) # B, T, C @ C, head_size -> B, T, H (head_size)
q = query(x) # B, T, C @ C, H -> B, T, H

# That's why a matric multiply of Key & Query will give higher
# affinities to tokens which are what other tokens are looking for
# Eg: a vowel is looking for certain consonants in positions before it
# Therefore, instead of starting weights with equal 0s, we now start
# with certain affinities we got from keys and queries
wei = q @ k.transpose(-2, -1) # (B, T, H) * (B, H, T) -> (B, T, T)

# Considering k and q are fairly gaussian distributions with variance = 1
# when we multiply them, the wei variance is roughly equal to the head size H
# Considering we're flowing these weights into a softmax later on, we don't
# want a high variance for wei as the exponentiation operation will give
# much higher probabilities to the higher values, so we do a normalization
# operation on wei beforehand
wei = wei * head_size**-0.5

# We're still making sure that future tokens cannot communicate with
# past tokens using a decoder block
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)

# Instead of just multiplying the weights with x, we use
# another linear transformation of x so that we can have
# values
value = nn.Linear(C, head_size, bias=False) # C, H
v = value(x) # B, T, C @ C, H -> B, T, H

out = wei @ v # B, T, T @ B, T, H -> B, T, H