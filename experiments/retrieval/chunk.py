import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Adds the parent directory (root) to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

CORPUS = Path(__file__).resolve().parent / "corpus"

from prompting.client import complete

system_prompt = """You are a teacher’s assistant for ICSE class 7. Use simple English. Do not invent a page number. If you are not sure, say so. Max 120 words."""

def answer(question: str, hit) -> dict:
    score, name, chunk = hit
    user = f"Source ({name}, overlap {score:.2f}):\n{chunk}\n\nQuestion: {question}\nAnswer only from the source. If it does not say, say you are not sure."
    reply = complete(system_prompt, user)
    return {"file": name, "score": score, "reply": reply}

def get_whole_chunk(content):
    return [
        (
            'WHOLE',
            content
        )
    ]

def get_fixed_window_chunks(content, window_size=80, overlap=20):
    effective_size = (window_size-overlap)
    total_windows = int(len(content)/effective_size) + 1
    chunks = []
    for i in range(total_windows):
        chunks.append(
            (
                f"FIXED_WINDOW_{i}",
                content[i*effective_size:(i*effective_size)+window_size]
            )
        )
    return chunks

def get_section_chunks(content, delimiter=r"\# "):
    chunks = []
    for i, chunk in enumerate(re.split(delimiter, content)):
        chunks.append(
            (
                f"SECTION_{i}",
                chunk
            )
        )
    return chunks

model = SentenceTransformer("all-MiniLM-L6-v2")
corpus_content = open(CORPUS/"photosynthesis_chapter.md", "r").read()

whole_file_chunks = get_whole_chunk(corpus_content)
fixed_window_chunks = get_fixed_window_chunks(corpus_content)
section_chunks = get_section_chunks(corpus_content)

def embed_search(question: str, chunks: list[tuple[str, str]], k: int = 3):
    bodies = [c[1] for c in chunks]
    doc_vecs = model.encode(bodies, normalize_embeddings=True)
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vecs @ q  # cosine, because both sides are unit length
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), chunks[i][0], chunks[i][1]) for i in order]

def search_print(question: str, complete: bool = False) -> None:
    hits = [(h[0], h[1]) for h in embed_search(question, whole_file_chunks)]

    fixed_window_hits = embed_search(question, fixed_window_chunks)
    hits2 = [(h[0], h[1]) for h in fixed_window_hits]

    section_hits = embed_search(question, section_chunks)
    hits3 = [(h[0], h[1]) for h in section_hits]

    print(f"Q: {question}\nWhole chunks: {hits}\nFixed Window chunks:{hits2}\nSection chunks: {hits3}\n\n")

    if complete:
        print(answer(question, fixed_window_hits[0]))
        print(fixed_window_hits[0][2])
        print(answer(question, section_hits[0]))
        print(section_hits[0][2])


q1 = "Are there 2 main stages of photosynthesis?"
search_print(q1)

q2 = "Is chlorophyll responsible for a plant's green colour?"
search_print(q2, complete=False)

q3 = "Does most life on Earth depend on photosynthesis?"
search_print(q3)

q4 = """Plants make food using chrolophyll and photosynthesis only in the leaves, and only in the day. What should I correct, for class 7?"""
search_print(q4)