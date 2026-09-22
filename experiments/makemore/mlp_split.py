import torch
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt

WORDS = open('names.txt', 'r').read().splitlines()
assert len(WORDS) == 32033

chars = sorted(list(set(''.join(WORDS))))
num_chars = len(chars) + 1 # special '.' character

ch_to_i = {s: i+1 for i,s in enumerate(chars)}
ch_to_i['.'] = 0

i_to_ch = {i+1: s for i,s in enumerate(chars)}
i_to_ch[0] = '.'

# Slice the dataset here before we start if needed
words = WORDS
random.shuffle(words)

BLOCK_SIZE = 3 # Context length: how many chars do we look at before predicting the next char

def build_dataset(words):
    inputs = []
    outputs = []

    for word in words:
        # print(word)
        context = [0] * BLOCK_SIZE
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

    return inputs, outputs

# Splits - remaining will be test split
TRAINING_SPLIT = 0.8 # 80%
DEV_VALIDATION_SPLIT = 0.1 # 10%

training_words = words[:int(TRAINING_SPLIT*len(words))]
dev_words = words[int(TRAINING_SPLIT*len(words)):int((TRAINING_SPLIT+DEV_VALIDATION_SPLIT)*len(words))]
test_words = words[int((TRAINING_SPLIT+DEV_VALIDATION_SPLIT)*len(words)):]

assert len(training_words) + len(dev_words) + len(test_words) == 32033
# We can't actually assert uniqueness of words across splits because
# the dataset has duplicate words
# assert set(training_words) & set(dev_words) == set()

training_inputs, training_outputs = build_dataset(training_words)
dev_inputs, dev_outputs = build_dataset(dev_words)
test_inputs, test_outputs = build_dataset(test_words)

embedding_dimensions = 2
embedding_table = torch.randn(num_chars, embedding_dimensions)

hidden_layer_width = 100
weights_1 = torch.randn((embedding_dimensions * BLOCK_SIZE, hidden_layer_width))
biases_1 = torch.randn(hidden_layer_width)
weights_2 = torch.randn((hidden_layer_width, num_chars))
biases_2 = torch.randn(num_chars)

parameters = [embedding_table, weights_1, biases_1, weights_2, biases_2]
num_params = 0
for p in parameters:
    p.requires_grad = True
    num_params += p.numel()
print(f'{num_params=}')

NUM_TRAINING_RUNS = 500000
MINI_BATCH_SIZE = 32

# Learning Rate Discovery
LEARNING_RATE = 0.111
LEARNING_RATE_DISCOVERY = False
LEARNING_RATE_START = -3 # 10 ^ -3 = 0.001
LEARNING_RATE_END = 0 # 10 ^ 0 = 1
learning_rate_exponents = torch.linspace(LEARNING_RATE_START, LEARNING_RATE_END, NUM_TRAINING_RUNS)
learning_rates = 10**learning_rate_exponents
learning_rate_i = []
loss_i = []

for i in range(NUM_TRAINING_RUNS):
    # MINIBATCH CONSTRUCT
    ix = torch.randint(0, training_inputs.shape[0], (MINI_BATCH_SIZE,))

    # FORWARD PASS
    embeddings = embedding_table[training_inputs[ix]]
    # torch.cat(torch.unbind(embeddings, 1), 1) @ weights_1 + biases_1
    # embeddings.view(embeddings.shape[0], embedding_dimensions * BLOCK_SIZE) @ weights_1 + biases_1
    squashed_embeddings = embeddings.view(-1, embedding_dimensions * BLOCK_SIZE)
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
    cross_entropy_loss = F.cross_entropy(logits, training_outputs[ix])
    if i % 1 == 0:
        print(f'{cross_entropy_loss=}')

    # BACKWARD PASS
    for p in parameters:
        p.grad = None
    cross_entropy_loss.backward()

    # UPDATE
    if LEARNING_RATE_DISCOVERY:
        learning_rate = learning_rates[i]
    else:
        # Decay the learning rate towards the end of the run
        if i < 0.90 * NUM_TRAINING_RUNS:
            learning_rate = LEARNING_RATE
        else:
            learning_rate = LEARNING_RATE/10

    for p in parameters:
        p.data += -learning_rate * p.grad

    # Learning Rate Disovery
    if LEARNING_RATE_DISCOVERY:
        learning_rate_i.append(learning_rate)
        loss_i.append(cross_entropy_loss.item())


if LEARNING_RATE_DISCOVERY:
    plt.plot(learning_rate_i, loss_i)
    plt.show()

# Considering we're using mini-batches, let's print the final loss over the entire dataset here:
def get_final_loss(dataset_name, inputs, outputs):
    print(f'{dataset_name=}')
    embeddings = embedding_table[inputs]
    squashed_embeddings = embeddings.view(-1, embedding_dimensions * BLOCK_SIZE)
    hidden_layer = torch.tanh(
        squashed_embeddings @ weights_1 + biases_1
    )
    logits = hidden_layer @ weights_2 + biases_2 # log counts
    loss = F.cross_entropy(logits, outputs)
    print(f'{loss=}')

get_final_loss('Training', training_inputs, training_outputs)
get_final_loss('Validation', dev_inputs, dev_outputs)
get_final_loss('Test', test_inputs, test_outputs)

# For a 2 dimensional embedding table, we can generate a scatter plot to see
# how similar characters are placed spatially together and special characters
# are isolated spatially
# plt.figure(figsize=(8,8))
# plt.scatter(embedding_table[:,0].data, embedding_table[:,1].data, s=200)
# for i in range(embedding_table.shape[0]):
#     plt.text(embedding_table[i,0].item(), embedding_table[i,1].item(), i_to_ch[i], ha="center", va="center", color="white")
# plt.grid('minor')
# plt.show()

# Sampling
for i in range(20):
    out = []
    ix = [0] * BLOCK_SIZE

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

# Argmax
# print('ARGMAX NAMES')
# for i in range(0, num_chars):
#     ix = [0] * (BLOCK_SIZE-1) + [i]
#     out = [i_to_ch[i]]

#     while True:
#         input_encoded = embedding_table[ix].view(1, -1)
#         hidden_layer = torch.tanh(input_encoded @ weights_1 + biases_1)
#         logits = hidden_layer @ weights_2 + biases_2
#         output_idx = torch.argmax(logits).item()
#         out.append(i_to_ch[output_idx])

#         if output_idx == 0:
#             break

#         ix = ix[1:] + [output_idx]

#     print(''.join(out))