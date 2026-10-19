from datasets import load_dataset

ds = load_dataset("rajpurkar/squad", split="validation")
print(len(ds))
print(ds[0].keys())
print(ds[0]["question"])
print(ds[0]["context"][:400])
print(ds[0]["answers"])

contexts = list(dict.fromkeys(ds["context"]))
print("questions", len(ds), "unique contexts", len(contexts))

index = {c: i for i, c in enumerate(contexts)}
gold_ids = [index[row["context"]] for row in ds]

# Random spot check
import random
r_index = random.randint(0, len(ds) - 1)
assert gold_ids[r_index] == index[ds[r_index]["context"]]

from sentence_transformers import SentenceTransformer
import numpy as np

# Embed all of the corpus
model = SentenceTransformer("all-MiniLM-L6-v2")
# Shape (n_contexts, n_embeddings)
doc = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)

# Embed all of the questions
# Shape (n_questions, n_embeddings)
q = model.encode(ds["question"], normalize_embeddings=True, show_progress_bar=True)

# Get cosine similarities
# Shape (n_questions, n_embeddings) @ (n_embeddins, n_context)
# Shape: n_questions × n_contexts
scores = q @ doc.T

# Did we get the correct context as the first hit of
# our embedding search?
top1 = scores.argmax(axis=1)
hit1 = float(np.mean(top1 == np.array(gold_ids)))
print(hit1)

# Partition the scores to get the index of the top 5 hits
# and then just slice the matrix to get the top 5 hits only
top5 = np.argpartition(scores, -5, axis=1)[:, -5:]
# For each question, we get the top1 context ID
# and check if that ID is in the top5 list
hit5 = float(np.mean([gold_ids[i] in top5[i] for i in range(len(gold_ids))]))
print(hit5)

# Print misses
# misses = np.where(top1 != np.array(gold_ids))[0][:5]
# for i in misses:
#     print("Q: ", ds[i]["question"])
#     print("gold answer: ", ds[i]["answers"]["text"][0])
#     print("ranked: ", contexts[top1[i]][:240])
#     print("gold: ", ds[i]["context"][:240])
#     print("---")

# Print hit5 misses
misses = [idx for idx, gold_context_id in enumerate(gold_ids) if gold_context_id not in top5[idx]][:5]
for i in misses:
    print("Q: ", ds[i]["question"])
    print("gold answer: ", ds[i]["answers"]["text"][0])
    print("ranked: ", contexts[top1[i]][:240])
    print("gold: ", ds[i]["context"][:240])
    print("---")