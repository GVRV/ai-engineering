import sys
import json

from agent import ask_model
from judge import judge_reply_grounding


def load_data(file_path):
    data = []

    f = open(file_path, 'r')
    for l in f.readlines():
        data.append(json.loads(l))

    return data


def calculate_calls(messages):
    tool_calls = 0

    for m in messages:
        try:
            if m.tool_calls:
                tool_calls += 1
        except AttributeError:
            pass

    return tool_calls


def get_tool_calls(messages):
    tool_calls = []

    for m in messages:
        try:
            if m["role"] == "tool":
                tool_calls.append(m["name"])
        except (TypeError, KeyError):
            pass

    return tool_calls

def get_expected_tool_call(kind):
    expected_tool_call = None

    if kind == "fact":
        expected_tool_call = "retrieve"
    elif kind == "headline":
        expected_tool_call = "classify"

    return expected_tool_call

def assert_tool_calls(tool_calls, expected_tool_call):
    for tool_call in tool_calls:
        if tool_call != expected_tool_call:
            return False

    return True

def get_first_query(messages):
    for m in messages:
        if m.tool_calls:
            for call in m.tool_calls:
                if call.function.name == "retrieve" and call.function.arguments:
                    args = json.loads(call.function.arguments)
                    return args["query"]


def calculate_grounding_judge(reply_data):
    reply = reply_data["reply"]
    paste_context = ""
    for m in reply_data["paste"]:
        if "content" in m:
            paste_context += m["content"]
    judgement = judge_reply_grounding(paste_context, reply)
    return judgement.startswith("YES;")


def print_stats(data):
    print("Total questions:", len(data))
    print("Total stopped:", len([d for d in data if d["stopped"]]))
    print("Total correct tool calls:", len([d for d in data if d["made_correct_tool_calls"] == True]))
    print("Total incorrect tool calls:", len([d for d in data if d["made_correct_tool_calls"] == False]))


def main():
    try:
        file_path = sys.argv[1]
    except IndexError:
        file_path = None

    if not file_path:
        print("usage: python eval_tools.py <file_path>")
        return

    data = load_data(file_path)
    f = open('runs/eval_day34.jsonl', 'w')

    for d in data:
        reply_data = ask_model(d['question'])
        tool_calls = get_tool_calls(reply_data["paste"])

        expected_tool_call = get_expected_tool_call(d["kind"])
        made_correct_tool_calls = assert_tool_calls(tool_calls, expected_tool_call)

        d["reply"] = reply_data["reply"]

        d["calls"] = calculate_calls(reply_data["paste"])
        d["tool_calls"] = tool_calls
        d["expected_tool_call"] = expected_tool_call
        d["made_correct_tool_calls"] = made_correct_tool_calls

        d["stopped"] = reply_data["reply"].startswith("STOPPED: model tried to call a tool after the cap")
        if d["stopped"]:
            d["reply"] = None

        if expected_tool_call == "retrieve" and made_correct_tool_calls:
            d["judge"] = None
            if not d["stopped"]:
                d["judge"] = calculate_grounding_judge(reply_data)
                d["first_query"] = get_first_query(reply_data["paste"])

        f.write(json.dumps(d))
        f.write("\n")

    print_stats(data)
    f.close()


if __name__ == "__main__":
    main()