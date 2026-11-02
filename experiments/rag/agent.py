# Adds the parent directory (root) to the Python path
import sys
from datasets import load_dataset
from pathlib import Path
from sentence_transformers import SentenceTransformer
import numpy as np
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
from prompting.client import complete_with_tool_calls

# Load the dataset
ds = load_dataset("rajpurkar/squad", split="validation")

contexts = list(dict.fromkeys(ds["context"]))

index = {c: i for i, c in enumerate(contexts)}
gold_ids = [index[row["context"]] for row in ds]

# Embed all of the corpus
model = SentenceTransformer("all-MiniLM-L6-v2")
# Shape (n_contexts, n_embeddings)
doc = model.encode(contexts, normalize_embeddings=True, show_progress_bar=True)

def retrieve(query: str = "") -> str:
    # rank against the same contexts as retrieve_squad.py
    # Embed the questions
    # Shape (1, n_embeddings)
    q = model.encode([query], normalize_embeddings=True)

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

    return paste_context

# Prompting stuff
SYSTEM_PROMPT = """You may call retrieve to look up facts. Answer only from paragraphs the tool returned. If they do not say, reply NOT_IN_SOURCE. Do not use outside knowledge to derive any facts, use multiple successive tool calls, if needed to determine your answer.
"""

def get_user_prompt(question):
    return f"""Question:
{question}
"""

TOOLS = [{
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": "Search the Wikipedia paragraph corpus. Returns the single most similar paragraph.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to look up."}
            },
            "required": ["query"],
        },
    },
}]

TOOL_FUNCS = {
    "retrieve": retrieve,
}

def ask_model(question: str) -> dict:
    extra_messages = []
    f = open('/tmp/agent.jsonl', 'w')

    # Hard-limit 3 tool calls
    for step in range(3):
        user_prompt = get_user_prompt(question)
        response = complete_with_tool_calls(
            SYSTEM_PROMPT,
            user_prompt,
            extra_messages=extra_messages,
            tools=TOOLS
        )

        if not response.tool_calls:
            break

        for call in response.tool_calls:
            call_id = call.id
            func_name = call.function.name
            func_args = call.function.arguments

            try:
                func_args = json.loads(call.function.arguments)
            except Exception:
                result = "ERROR_CALLING_TOOL: Arguments are not valid JSON"
            else:
                result = TOOL_FUNCS[func_name](**func_args)

            f.write(json.dumps({
                "call": call_id,
                "name": func_name,
                "args": func_args,
            }))
            f.write("\n")

            new_message = {
                "role": "tool",
                "tool_call_id": call_id,
                "content": result,
                "name": func_name,
            }
            f.write(json.dumps(new_message))
            f.write("\n")
            extra_messages.append(new_message)

    f.close()

    final_response = complete_with_tool_calls(
        SYSTEM_PROMPT,
        user_prompt,
        extra_messages=extra_messages,
        tools=None,
    )

    return {
        "reply": response.content,
        "paste": extra_messages,
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