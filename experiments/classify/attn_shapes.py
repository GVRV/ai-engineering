"""
One AG News sentence through DistilBERT layer 0.
Prints Q, K, V shapes and the attention weights for the first head.
"""

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModel.from_pretrained("distilbert-base-uncased")
model.eval()

# one real sentence
ds = load_dataset("fancyzhx/ag_news", split="test")
text = ds[0]["text"]
print("SENTENCE:", text[:300])

inputs = tokenizer(text, return_tensors="pt", truncation=True)
# input_ids: (1, seq)
seq = inputs["input_ids"].shape[1]
print("seq:", seq)

# embeddings that enter the transformer, shape (1, seq, 768)
with torch.no_grad():
    hidden = model.embeddings(inputs["input_ids"])

layer = model.transformer.layer[0]
# these three linears are W_Q, W_K, W_V
Q = layer.attention.q_lin(hidden)  # (1, seq, 768)
K = layer.attention.k_lin(hidden)
V = layer.attention.v_lin(hidden)
print("Q:", tuple(Q.shape), "K:", tuple(K.shape), "V:", tuple(V.shape))

hidden_size = model.config.hidden_size # Distilbert has 768 dimensions
num_attn_heads = model.config.num_attention_heads # Distilbert has 12 heads
head_size = hidden_size // num_attn_heads # 64 dimensions

INSPECT_HEAD_NUMBER = 5
# first head only: 64 of the 768 dims
# DistilBERT has 12 heads, 768 / 12 = 64
q = Q[0, :, head_size*(INSPECT_HEAD_NUMBER-1):head_size*INSPECT_HEAD_NUMBER]   # (seq, 64)
k = K[0, :, head_size*(INSPECT_HEAD_NUMBER-1):head_size*INSPECT_HEAD_NUMBER]
scale = head_size ** 0.5  # As the dot product will increase large numbers to even larger values, we scale back before using softmax
scores = q @ k.T / scale          # (seq, seq)
weights = torch.softmax(scores, dim=-1)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
print("TOKENS:", tokens)
i = 1  # token 1 is usually the first real word (token 0 is [CLS])
print(f"attention from '{tokens[i]}':")
for j, w in enumerate(weights[i].tolist()):
    print(f"  {w:0.3f}  {tokens[j]}")

v = V[0, :, head_size*(INSPECT_HEAD_NUMBER-1):head_size*INSPECT_HEAD_NUMBER]                 # (seq, 64), first head
mixed = weights @ v              # (seq, seq) @ (seq, 64) = (seq, 64)
print("mixed shape:", tuple(mixed.shape))