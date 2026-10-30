# Adds the parent directory (root) to the Python path
import sys
from datasets import load_dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from prompting.client import complete

# Prompting stuff
SYSTEM_PROMPT = """Answer only from the paragraph. If the paragraph does not say, reply exactly `NOT_IN_SOURCE`. Do not use outside knowledge.
"""

def get_user_prompt(paragraph, question):
    return f"""Paragraph:
{paragraph}

Question:
{question}
"""

# Load the dataset
ds = load_dataset("rajpurkar/squad", split="validation")

contexts = list(dict.fromkeys(ds["context"]))

index = {c: i for i, c in enumerate(contexts)}
gold_ids = [index[row["context"]] for row in ds]

# Embed all of the corpus
model = SentenceTransformer("all-MiniLM-L6-v2")
# Shape (n_contexts, n_embeddings)
doc = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)

def ask_model(question: str) -> dict:
    # rank against the same contexts as retrieve_squad.py
    # Embed the questions
    # Shape (1, n_embeddings)
    q = model.encode([question], normalize_embeddings=True)

    # Get cosine similarities
    # Shape (1, n_embeddings) @ (n_embeddins, n_context)
    # Shape: 1 × n_contexts
    scores = q @ doc.T

    # Did we get the correct context as the first hit of
    # our embedding search?
    top1 = scores.argmax(axis=1)[0]
    top1_score = scores.max(axis=1)[0]

    # paste contexts[top1][:1500]
    context = contexts[top1]
    paste_context = context[:1500]

    user_prompt = get_user_prompt(paste_context, question)
    response = complete(SYSTEM_PROMPT, user_prompt)

    return {
        "reply": response,
        "paste": paste_context,
        "top_score": float(top1_score),
    }

def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("usage: python ask.py <question>")
        return

    data = ask_model(question)

    # print reply, then the paste
    print("Answer:", data["reply"])
    print("Context:", data["paste"])


if __name__ == "__main__":
    main()