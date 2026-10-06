import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

def complete(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # if this id 404s, print client.models.list() and pick a chat model
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content