import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

USAGE = {
    'completion_tokens': 0,
    'prompt_tokens': 0,
    'total_tokens': 0,
}

def update_usage(usage):
    USAGE['completion_tokens'] += usage.completion_tokens
    USAGE['prompt_tokens'] += usage.prompt_tokens
    USAGE['total_tokens'] += usage.total_tokens

def get_usage():
    return USAGE.copy()

def clear_usage():
    USAGE['completion_tokens'] = 0
    USAGE['prompt_tokens'] = 0
    USAGE['total_tokens'] = 0


def complete_with_tool_calls(system: str, user: str, extra_messages=None, tools=None) -> str:
    messages=[
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if extra_messages:
        messages += extra_messages

    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # if this id 404s, print client.models.list() and pick a chat model
        messages=messages,
        temperature=0,
        tools=tools,
    )
    update_usage(resp.usage)
    return resp.choices[0].message


def complete(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model="openai/gpt-oss-20b",  # if this id 404s, print client.models.list() and pick a chat model
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0,
    )
    update_usage(resp.usage)
    return resp.choices[0].message.content