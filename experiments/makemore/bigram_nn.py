import torch
import torch.nn.functional as F
import random

WORDS = open('names.txt', 'r').read().splitlines()

chars = sorted(list(set(''.join(WORDS))))
num_chars = len(chars) + 1 # special '.' character

ch_to_i = {s: i+1 for i,s in enumerate(chars)}
ch_to_i['.'] = 0

i_to_ch = {i+1: s for i,s in enumerate(chars)}
i_to_ch[0] = '.'

inputs = []
outputs = []

for word in WORDS:
    chars = ['.'] + list(word) + ['.']
    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = ch_to_i[ch1]
        ix2 = ch_to_i[ch2]

        inputs.append(ix1)
        outputs.append(ix2)

assert len(inputs) == len(outputs)

inputs_encoded = F.one_hot(torch.tensor(inputs), num_classes=num_chars).float()

assert inputs_encoded.dtype == torch.float32
assert inputs_encoded.shape[0] == 228146 # Number of predictions
assert inputs_encoded.shape[1] == num_chars # 27

# Random sample check
i = random.randint(0, 228146)
# Only one value is 1
assert inputs_encoded[i].sum() == 1
# The index of the value which is 1 is the index of the input character
assert inputs_encoded[i][inputs[i]].item() == 1

# initialise weights randomly
# This is a 27 * 27 matrix
# This is a gaussian distribution with 0 as mean and variance = 1
weights = torch.randn((num_chars, num_chars), requires_grad=True)

log_counts = inputs_encoded @ weights # logits

# Softmax
counts = log_counts.exp() # similar to our counts in the previous model
probabilities = counts / counts.sum(1, keepdims=True)

# sanity check
assert probabilities[i].sum().item() - 1 < 0.00001

num_samples = 0
cumulative_loss = 0
for idx, output in enumerate(outputs):
    # Probabilities our model predicts
    predicted_probabilities = probabilities[idx]

    # Probability of the ground truth as per our model
    predicted_probability = predicted_probabilities[output]
    # print(f'{predicted_probability=}')

    # Log Likelikehood
    log_likelihood = predicted_probability.log()
    # print(f'{log_likelihood=}')

    # Negative Log Likelikehood
    neg_log_likelihood = -log_likelihood
    # print(f'{neg_log_likelihood=}')

    num_samples += 1
    cumulative_loss += neg_log_likelihood

print(f'{num_samples=}')
print(f'Average loss: {cumulative_loss/num_samples}')

# learning loop
for _ in range(500):
    # forward pass
    log_counts = inputs_encoded @ weights # logits
    counts = log_counts.exp() # similar to our counts in the previous model
    probabilities = counts / counts.sum(1, keepdims=True)
    # Get the output index value from the range 0->end of each row
    p_loss = -probabilities[torch.arange(len(outputs)), outputs].log().mean()
    # Regularization loss: making the probabilities more uniform
    # similar to adding +1 to the counts in the previous example)
    # If this loss dominates the previous one, we get much more uniform
    # predictions instead of ground truth
    r_loss = 0.01 * (weights**2).mean()
    loss = p_loss + r_loss

    # backward pass
    weights.grad = None
    loss.backward()

    if _ % 10 == 0:
        print(f'{loss=}')

    # update
    weights.data += -50 * weights.grad

# Sampling
for i in range(20):
    out = []
    ix = 0

    while True:
        input_encoded = F.one_hot(torch.tensor([ix]), num_classes=num_chars).float()
        logits = input_encoded @ weights
        counts = logits.exp()
        p = counts / counts.sum(1, keepdims=True)
        ix = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(i_to_ch[ix])

        if ix == 0:
            break

    print(''.join(out))