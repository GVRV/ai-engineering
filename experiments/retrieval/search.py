import re
import sys
from pathlib import Path

# Adds the parent directory (root) to the Python path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from prompting.client import complete

def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def search(question: str, k: int = 3) -> list[tuple[float, str, str]]:
    q = tokens(question)
    hits = []
    for path in sorted(Path("corpus").glob("*.txt")):
        body = path.read_text()
        t = tokens(body)
        score = len(q & t) / max(len(q), 1)
        hits.append((score, path.name, body))
    hits.sort(key=lambda row: row[0], reverse=True)
    return hits[:k]

def search_print(question: str) -> None:
    hits = [(h[0], h[1]) for h in search(question)]
    print(f"{question} --> \n\n{hits}")


q1 = "Are there 2 main stages of photosynthesis?"
search_print(q1)
q2 = "Is the potato plant the best at photosynthesis?"
search_print(q2)
q3 = "Is the stomach the most important organ for digestion?"
search_print(q3)

question = """Plants make food using chrolophyll and photosynthesis only in the leaves, and only in the day. What should I correct, for class 7?"""
search_print(question)

system_prompt = """You are a teacher’s assistant for ICSE class 7. Use simple English. Do not invent a page number. If you are not sure, say so. Max 120 words."""

def answer(question: str) -> dict:
    hits = search(question, k=1)
    score, name, chunk = hits[0]
    user = f"Source ({name}, overlap {score:.2f}):\n{chunk}\n\nQuestion: {question}\nAnswer only from the source. If it does not say, say you are not sure."
    reply = complete(system_prompt, user)
    return {"file": name, "score": score, "reply": reply}

print(answer(question))

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
    print(q)
    print(answer(q))