# Adds the parent directory (root) to the Python path
import sys
from pathlib import Path
import json

sys.path.append(str(Path(__file__).resolve().parents[1]))
from prompting.client import complete

# Prompting stuff
SYSTEM_PROMPT = """You grade whether a reply is supported by a paragraph.
Supported means every factual claim in the reply is stated in the paragraph.
A refusal of NOT_IN_SOURCE is supported when the paragraph does not contain the answer.
A refusal is not supported when the paragraph does contain it.
Reply with one line: YES or NO, then a semicolon, then a reason under 20 words.
Do not use knowledge from outside the paragraph.
"""

def get_user_prompt(paste, model_reply):
    return f"""Paragraph:
{paste}

Reply:
{model_reply}
"""

def judge_reply_grounding(paste: str, reply: str) -> str:
    user_prompt = get_user_prompt(paste, reply)
    return complete(SYSTEM_PROMPT, user_prompt)


def main():
    judging_file_path = " ".join(sys.argv[1:]).strip()
    if not judging_file_path:
        print("usage: python judge.py <path_to_json_file>")
        return

    with open(judging_file_path, 'r') as f:
        for json_line in f:
            data = json.loads(json_line.strip())

            paste = data["paste"]
            reply = data["reply"]

            reply_judgement = judge_reply_grounding(
                paste, reply
            )

            print("----")
            print("Squad Question:", data["question"])
            print("Context:", paste)
            print("Model Answer:", reply)
            print("Judge:", reply_judgement)
            print("----")



if __name__ == "__main__":
    main()