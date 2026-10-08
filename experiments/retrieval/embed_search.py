import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
from fastembed import TextEmbedding
import numpy as np

# Adds the parent directory (root) to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

CORPUS = Path(__file__).resolve().parent / "corpus"

from prompting.client import complete

STOP = {"a", "an", "the", "in", "on", "of", "for", "to", "and", "or",
        "is", "are", "was", "what", "should", "only", "class", "day"}

def tokens(text: str) -> set[str]:
    return set(t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP)

def search(question: str, k: int = 3) -> list[tuple[float, str, str]]:
    q = tokens(question)
    hits = []
    for path in sorted(CORPUS.glob("*.txt")):
        body = path.read_text()
        t = tokens(body)
        score = len(q & t) / max(len(q), 1)
        hits.append((score, path.name, body))
    hits.sort(key=lambda row: row[0], reverse=True)
    return hits[:k]

model = SentenceTransformer("all-MiniLM-L6-v2")
files = sorted(CORPUS.glob("*.txt"))
bodies = [p.read_text() for p in files]
doc_vecs = model.encode(bodies, normalize_embeddings=True)

def embed_search(question: str, k: int = 3):
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vecs @ q  # cosine, because both sides are unit length
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), files[i].name) for i in order]

model_2 = TextEmbedding()
doc_vecs_2 = np.array(list(model_2.embed(bodies)))
doc_vecs_2_norm = np.linalg.norm(doc_vecs_2, axis=1, keepdims=True)

def embed_search_2(question: str, k: int = 3):
    q = np.array(list(model_2.embed([question])))
    q_norm = np.linalg.norm(q, axis=1, keepdims=True)
    scores = ((q @ doc_vecs_2.T) / (q_norm @ doc_vecs_2_norm.T))[0]
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), files[i].name) for i in order]

def search_print(question: str) -> None:
    hits = [(h[0], h[1]) for h in embed_search(question)]
    hits2 = [(h[0], h[1]) for h in embed_search_2(question)]
    search_hits = [(h[0], h[1]) for h in search(question)]
    print(f"Q: {question} --> \nEmbeddings: {hits}\nFast Embed:{hits2}\nStopword Overlap: {search_hits}\n\n")


q1 = "Are there 2 main stages of photosynthesis?"
search_print(q1)
q2 = "Is the potato plant the best at photosynthesis?"
search_print(q2)
q3 = "Is the stomach the most important organ for digestion?"
search_print(q3)

question = """Plants make food using chrolophyll and photosynthesis only in the leaves, and only in the day. What should I correct, for class 7?"""

system_prompt = """You are a teacher’s assistant for ICSE class 7. Use simple English. Do not invent a page number. If you are not sure, say so. Max 120 words."""

# def answer(question: str) -> dict:
#     hits = embed_search(question, k=1)
#     score, name, chunk = hits[0]
#     user = f"Source ({name}, overlap {score:.2f}):\n{chunk}\n\nQuestion: {question}\nAnswer only from the source. If it does not say, say you are not sure."
#     reply = complete(system_prompt, user)
#     return {"file": name, "score": score, "reply": reply}

search_print(question)

QUESTIONS = [
    "Are the chemicals involved in photosynthesis carbon dioxide and oxygen?",
    "Is photosynthesis less effective during winter months?", # TRICK
    "Is chlorophyll responsible for a plant's green colour?",
    "Are fruits where plant stores the energy produced by photosynthesis?", # TRICK
    "Does most life on Earth depend on photosynthesis?",
    "Are school grammar classes voluntary for students to attend?",
    "Did Gandhi's plant based diet give him the energy to fight for independence?",
    "Plants are green because that colour helps with plant reproduction?",
    "The water cycle is essential for all plants to survive. True or False?",
    "Who was the first human to land on the moon?",
]

for q in QUESTIONS:
    # print(q)
    search_print(q)