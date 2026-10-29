# Adds the parent directory (root) to the Python path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from prompting.client import complete
import re
import json

SYSTEM_PROMPT = """Answer only from the paragraph. If the paragraph does not say, reply exactly `NOT_IN_SOURCE`. Do not use outside knowledge.
"""
def get_user_prompt(paragraph, question):
    return f"""Paragraph:
{paragraph}

Question:
{question}
"""

from datasets import load_dataset
ds = load_dataset("rajpurkar/squad", split="validation")

contexts = list(dict.fromkeys(ds["context"]))

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

rng = np.random.default_rng(42)
hits = np.where(top1 == np.array(gold_ids))[0]
misses = np.where(top1 != np.array(gold_ids))[0]
chosen = np.concatenate([
    rng.choice(hits, size=10, replace=False),
    rng.choice(misses, size=10, replace=False),
])

def norm(s):
    s = s.lower()
    s = s.replace(" and ", " ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())

def contains(paste, answers):
    """Does any part of the answer word exist in the
    paste text which was sent to the model to generate
    an answer?
    """
    paste_n = norm(paste)
    return any(
        norm(a) in paste_n
        for a in answers if a.strip()
    )

f = open('/tmp/answer_log.jsonl', 'w')

for idx in chosen:
    context = contexts[top1[idx]]

    squad_question = ds[idx]["question"]
    squad_answer = ds[idx]["answers"]["text"]
    paste_context = context[:1500]
    gold_hit = bool(top1[idx] == gold_ids[idx])
    answer_in_paste = contains(paste_context, squad_answer)

    user_prompt = get_user_prompt(paste_context, squad_question)
    response = complete(SYSTEM_PROMPT, user_prompt)


    print("----")
    print("Squad Question:", squad_question)
    print("Context:", paste_context)
    print("Squad Answer:", squad_answer)
    print("Model Answer:", response)
    print("Gold hit:", gold_hit)
    print("Answer in Paste:", answer_in_paste)
    print("----")

    judging_data = {
        "index": int(idx),
        "question": squad_question,
        "paste": paste_context,
        "reply": response,
        "gold_hit": gold_hit,
        "answer_in_paste_normalized": answer_in_paste,
    }

    f.write(json.dumps(judging_data))
    f.write("\n")

f.close()
