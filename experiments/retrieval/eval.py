import re
import sys
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np

# Adds the parent directory (root) to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

CORPUS = Path(__file__).resolve().parent / "corpus"

def get_whole_chunk(content):
    return [
        (
            'WHOLE',
            content
        )
    ]

def get_fixed_window_chunks(content, window_size=80, overlap=20):
    words = content.split()
    chunks = []
    for i in range(0, len(words) - window_size + 1, window_size - overlap):
        chunk = " ".join(words[i: i + window_size])
        chunks.append(
            (
                f"FIXED_WINDOW_{len(chunks)}",
                chunk
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

import pdb; pdb.set_trace()

def embed_search(question: str, chunks: list[tuple[str, str]], k: int = 3):
    bodies = [c[1] for c in chunks]
    doc_vecs = model.encode(bodies, normalize_embeddings=True)
    q = model.encode([question], normalize_embeddings=True)[0]
    scores = doc_vecs @ q  # cosine, because both sides are unit length
    order = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), chunks[i][0], chunks[i][1]) for i in order]

def search_print(question: str, expected_section_id: int = None) -> None:
    hits = [(h[0], h[1]) for h in embed_search(question, whole_file_chunks)]

    fixed_window_hits = embed_search(question, fixed_window_chunks)
    hits2 = [(h[0], h[1]) for h in fixed_window_hits]

    section_hits = embed_search(question, section_chunks)
    hits3 = [(h[0], h[1]) for h in section_hits]

    print(f"question: {question}")
    print(f"expected section id, or none: {expected_section_id}")

    section_id = section_hits[0][1]
    print(f"top section id: {section_id}")

    fixed_window_id = fixed_window_hits[0][1]
    print(f"top window id: {fixed_window_id}")

    assertion = f"SECTION_{expected_section_id}" == section_id
    print(f"hit: {assertion}\n")

# (question, expected section) tuple
QUESTIONS = [
    ("Are the chemicals involved in photosynthesis carbon dioxide and oxygen?", 2),
    ("Is photosynthesis less effective during winter months?", None),
    ("Is chlorophyll responsible for a plant's green colour?", 3),
    ("Are fruits where plant stores the energy produced by photosynthesis?", 6),
    ("Does most life on Earth depend on photosynthesis?", 1),
    ("Are school grammar classes voluntary for students to attend?", None),
    ("Did Gandhi's plant based diet give him the energy to fight for independence?", None),
    ("Plants are green because that colour helps with plant reproduction?", None),
    ("The water cycle is essential for all plants to survive. True or False?", 1),
    ("Who was the first human to land on the moon?", None),
]

for q in QUESTIONS:
    search_print(*q)