import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

# Don't have cuda :(
# print(torch.cuda.is_available())

TEXT = open('input.txt', 'r', encoding='utf-8').read()
# print(len(TEXT))

TOKENS = sorted(list(set(TEXT))) # unique characters in the dataset
VOCAB_SIZE = len(TOKENS)
NUM_EMBEDDING = 32
NUM_TRANSFORMERS = 3
DROPOUT_PROB = 0.2
TRAINING_ITERS = 5000
NUM_HEADS = 4
EVAL_ITERATIONS = 200
# print(len(TOKENS))
# print(TOKENS)

stoi = {ch: i for i,ch in enumerate(TOKENS)}
itos = {i: ch for i,ch in enumerate(TOKENS)}
# Take a string and encode it into a list of token identifiers
encode = lambda str_text: [stoi[ch] for ch in str_text]
# Take a list of token identifiers and decode it into a string
decode = lambda tokens: "".join([itos[token] for token in tokens])
# print(encode("hii there"))
assert decode(encode("hii there")) == "hii there"

data = torch.tensor(encode(TEXT), dtype=torch.long)
# print(data.shape)
# print(data[:100])

# Split into training and validation datasets
n = int(len(data) * 0.90)
train_data = data[:n]
val_data = data[n:]

# When we want to train on the data, we want to train the network
# to make predictions over a certain number of tokens
# i.e. Given $X number of tokens, what character will follow?
# This is called the block_size or context_size
block_size = 8

# For GPU efficiency, we again use a bunch of independent block sizes
# while training so that calculations can be done in parallel faster
# The number of datasets being trained on in parallel is called
# the batch_size
batch_size = 32

def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb, yb = get_batch('train')
assert xb.shape == torch.Size([batch_size, block_size])
assert yb.shape == torch.Size([batch_size, block_size])

# In a given sequence of `block_size` characters, there are
# actually `block_size` number of predictions and each block
# in the batch is training on its predictions
# for b in range(batch_size):
#     for t in range(block_size):
#         context = xb[b, :t+1]
#         target = yb[b, t]
#         print(f"when input is {context.tolist()} the target is: {target}")

class Head(nn.Module):
    """ one head of self-attention """
    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(NUM_EMBEDDING, head_size, bias=False)
        self.query = nn.Linear(NUM_EMBEDDING, head_size, bias=False)
        self.value = nn.Linear(NUM_EMBEDDING, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(DROPOUT_PROB)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x) # B, T, C
        q = self.query(x) # B, T, C

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # B, T, C @ B, C, T -> B, T, T
        # decoder block
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B, T, T
        wei = F.softmax(wei, dim=-1) # B, T, T
        wei = self.dropout(wei)

        v = self.value(x) # B, T, C
        out = wei @ v # B, T, T @ B, T, C -> B, T, C
        return out


class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, num_heads * head_size)
        self.dropout = nn.Dropout(DROPOUT_PROB)

    def forward(self, x):
        # concatenate head outputs in the Channel dimension
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # let them intermix through a linear layer
        out = self.projection(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity
    """
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width * 4),
            nn.ReLU(),
            nn.Linear(width * 4, width), # projection operation
            nn.Dropout(DROPOUT_PROB),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.sa_heads = MultiHeadAttention(num_heads, head_size)
        self.ffwd = FeedForward(NUM_EMBEDDING)
        self.ln_1 = nn.LayerNorm(NUM_EMBEDDING)
        self.ln_2 = nn.LayerNorm(NUM_EMBEDDING)

    def forward(self, x):
        x = x + self.sa_heads(self.ln_1(x)) # B, T, C
        x = x + self.ffwd(self.ln_2(x)) # B, T, C
        return x



class TransformerModel(nn.Module):

    def __init__(self):
        super().__init__()
        # for each token as a row index, we have the logits of all
        # of the other tokens that can follow it in the columns
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE, NUM_EMBEDDING)
        # We also want to encode where in the block size the input is (is it
        # the first character? in the middle? etc?)
        self.position_embedding_table = nn.Embedding(block_size, NUM_EMBEDDING)
        self.blocks = nn.Sequential(*[Block(NUM_HEADS, NUM_EMBEDDING//NUM_HEADS) for _ in range(NUM_TRANSFORMERS)])
        self.lm_head = nn.Linear(NUM_EMBEDDING, VOCAB_SIZE)
        self.ln = nn.LayerNorm(NUM_EMBEDDING)

    def forward(self, idx, targets=None):
        B,T = idx.shape

        # idx and targets are both (Batch,Time) tensor of integers
        token_embedding = self.token_embedding_table(idx) # (Batch,Time,Channel)
        position_embedding = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = token_embedding + position_embedding # Broadcasting: B, T, C
        x = self.blocks(x)
        logits = self.lm_head(self.ln(x)) # (Batch, Time, Vocab_Size)

        # If we're doing a forward pass without any target predictions
        # we have no loss to calculate
        if targets is None:
            loss = None
        else:
            # Because cross_entropy expects Channels as the second dimension
            # we need to reshape our logits and targets
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 2-dimensional, character -> logits for next prediction
            targets = targets.view(B*T) # 1-dimensional -> expected prediction

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            # we need to truncate the input for the forward pass to block size
            # so that T doesn't extend beyond what the embeddings can handle
            logits, loss = self(idx[:, -block_size:]) # logits without loss is (B, T, C)
            # focus on only the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # TODO: Can also be 1, right?
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # becomes (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

# Initialise the model
m = TransformerModel()
logits, loss = m(xb, yb)
# print(logits.shape)
# print(loss)

# A much better way to get the mean loss across batches across splits
@torch.no_grad()
def estimate_loss():
    out = {}

    # Inform Pytorch we're in evaluation mode
    m.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(EVAL_ITERATIONS)

        for k in range(EVAL_ITERATIONS):
            X, Y = get_batch(split)
            logits, loss = m(X, Y)
            losses[k] = loss.item()

        out[split] = losses.mean()

    # Inform Pytorch we're back in training mode
    m.train()

    return out

idx = torch.zeros((1, 1), dtype=torch.long)

# Sample from the model
def get_prediction(size=200):
    predictions = m.generate(idx, max_new_tokens=size) # (B, T)
    prediction = predictions[0] # (T)
    print(decode(prediction.tolist()))

# Training loop
# print(list(m.parameters()))
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
for steps in range(TRAINING_ITERS):
    xb, yb = get_batch('train')

    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 500 == 0:
        print(estimate_loss())

print(estimate_loss())
get_prediction()