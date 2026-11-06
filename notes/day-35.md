A routing score without a cost is half a number.

I will count tool calls and model calls per question on the Day 33 twenty and the Day 34 ten.

I will record prompt and completion tokens on one fresh run of the ten, because the old jsonl does not have them.

Cost is calls times tokens times the published price. I will name the price and the date I read it.

# Block A - calls I already have

## eval_day33.jsonl

- total questions: 20
- total tool calls: 33
    - 1 tool call: 12
    - 2 tool calls: 3
    - 3 tool calls (max): 5
- mean tool calls per question: 1.65
- how many `STOPPED`: 3

## eval_day34.jsonl

- total questions: 10
- total tool calls: 14
    - 1 tool call: 8
    - 2 tool calls: 0
    - 3 tool calls (max): 2
- mean tool calls per question: 1.4
- how many `STOPPED`: 2

# Block B

Total prompt tokens: 13856
Price per input tokens: $0.075 / 1 M
Total price for prompt tokens: $0.0010392

Total completion tokens: 2305
Price per output tokens: $.30 / 1 M
Total price for completion tokens: $0.0006915

Total Price: $0.0017307
Total Questions: 10
Mean Price per Question: $0.00017307


# Notes

Price source and date: https://console.groq.com/docs/models (2026-11-06) openai/gpt-oss-20b: $0.075 input/1M tokens; $0.30 output/1M tokens;
Mean tool calls, twenty: 1.65
Mean tool calls, ten: 1.4
Mean tokens per question, ten: 1616.1
Mean dollars per question: $0.00017307
Worst question: How much of Paris' population was killed by the plague? (3786 total tokens)

Usage missing: no
Did I invent a token count: no

What I still find shaky, in full sentences:
- Nothing about costs is shaky, makes sense (surprised by how cheap it is though, even though we're using an "old" model)