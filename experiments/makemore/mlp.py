import torch
import torch.nn.functional as F
import random

WORDS = open('names.txt', 'r').read().splitlines()
assert len(WORDS) == 32033

chars = sorted(list(set(''.join(WORDS))))
num_chars = len(chars) + 1 # special '.' character

ch_to_i = {s: i+1 for i,s in enumerate(chars)}
ch_to_i['.'] = 0

i_to_ch = {i+1: s for i,s in enumerate(chars)}
i_to_ch[0] = '.'

# Let's start with just the first eight
words = WORDS

block_size = 3 # Context length: how many chars do we look at before predicting the next char

inputs = []
outputs = []

for word in words:
    # print(word)
    context = [0] * block_size
    for ch in word + '.':
        idx = ch_to_i[ch]
        inputs.append(context)
        outputs.append(idx)

        readable_context = ''.join(i_to_ch[x] for x in context)
        # print(f'{readable_context} ({context}) -> {i_to_ch[idx]} ({idx})')

        context = context[1:] + [idx]

inputs = torch.tensor(inputs)
outputs = torch.tensor(outputs)

assert inputs.shape[1] == 3
assert inputs.shape[0] == outputs.shape[0]

embedding_dimensions = 2
embedding_table = torch.randn(num_chars, embedding_dimensions)

hidden_layer_width = 100
weights_1 = torch.randn((embedding_dimensions * block_size, hidden_layer_width))
biases_1 = torch.randn(hidden_layer_width)
weights_2 = torch.randn((hidden_layer_width, num_chars))
biases_2 = torch.randn(num_chars)

parameters = [embedding_table, weights_1, biases_1, weights_2, biases_2]
num_params = 0
for p in parameters:
    p.requires_grad = True
    num_params += p.numel()
print(f'{num_params=}')

NUM_TRAINING_RUNS = 50000
LEARNING_RATE = 0.09
MINI_BATCH_SIZE = 32
for _ in range(NUM_TRAINING_RUNS):
    # MINIBATCH CONSTRUCT
    ix = torch.randint(0, inputs.shape[0], (MINI_BATCH_SIZE,))

    # FORWARD PASS
    embeddings = embedding_table[inputs[ix]]
    # torch.cat(torch.unbind(embeddings, 1), 1) @ weights_1 + biases_1
    # embeddings.view(embeddings.shape[0], embedding_dimensions * block_size) @ weights_1 + biases_1
    squashed_embeddings = embeddings.view(-1, embedding_dimensions * block_size)
    hidden_layer = torch.tanh(
        squashed_embeddings @ weights_1 + biases_1
    )
    logits = hidden_layer @ weights_2 + biases_2 # log counts

    # counts = logits.exp()
    # probabilities = counts / counts.sum(1, keepdims=True)
    # # sanity check
    # assert abs(probabilities[0].sum() - 1) < 0.0001

    # loss = -probabilities[torch.arange(probabilities.shape[0]), outputs].log().mean()
    # print(f'{loss=}')

    # Cross Entropy loss is the same as the classification calculation we're doing above
    cross_entropy_loss = F.cross_entropy(logits, outputs[ix])
    if _ % 100 == 0:
        print(f'{cross_entropy_loss=}')

    # BACKWARD PASS
    for p in parameters:
        p.grad = None
    cross_entropy_loss.backward()

    # UPDATE
    for p in parameters:
        p.data += -LEARNING_RATE * p.grad

# Considering we're using mini-batches, let's print the final loss here:
embeddings = embedding_table[inputs]
squashed_embeddings = embeddings.view(-1, embedding_dimensions * block_size)
hidden_layer = torch.tanh(
    squashed_embeddings @ weights_1 + biases_1
)
logits = hidden_layer @ weights_2 + biases_2 # log counts
loss = F.cross_entropy(logits, outputs)
print(f'{loss=}')

# Sampling
for i in range(20):
    out = []
    ix = [0] * block_size

    while True:
        input_encoded = embedding_table[ix].view(1, -1)
        hidden_layer = torch.tanh(input_encoded @ weights_1 + biases_1)
        logits = hidden_layer @ weights_2 + biases_2
        # counts = logits.exp()
        # p = counts / counts.sum(1, keepdims=True)
        p = F.softmax(logits, dim=1)
        output_idx = torch.multinomial(p, num_samples=1, replacement=True).item()
        out.append(i_to_ch[output_idx])

        if output_idx == 0:
            break

        ix = ix[1:] + [output_idx]

    print(''.join(out))