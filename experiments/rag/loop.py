import sys
from ask import ask_model
from judge import judge_reply_grounding

def answer_with_judge(question: str) -> dict:
    data = ask_model(question)
    return {
        "reply": data["reply"], # model reply
        "paste": data["paste"], # paste context for the model
        "judge": judge_reply_grounding(data["paste"], data["reply"]), # model judge whether reply grounded
        "top_score": data["top_score"], # top cosine similarity score
    }

def main():
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print("usage: python ask.py <question>")
        return

    data = answer_with_judge(question)

    print("Model Reply:", data["reply"])
    print("Grounded Reply Judgement:", data["judge"])
    print("Paste Context:", data["paste"][:400])
    print("Top Cosine Score:", data["top_score"])

if __name__ == "__main__":
    main()