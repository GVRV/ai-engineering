The agent runs on my machine. A hiring manager cannot call it.

Today I put ask_model behind one HTTP route so a curl command returns the reply, the tool calls, and the judge line.

This is not a production service. It has no auth, no queue, and it loads the models on startup. It exists so the README has a command that works.

# Responses

```
$ curl -s localhost:8000/health
{"ok":true}
```

```
$ curl -s -X POST localhost:8000/ask \
    -H 'content-type: application/json' \
    -d '{"question": "How high is the highest point in Warsaw?"}'
{"reply":"The highest point in Warsaw is Szczęśliwice hill, which reaches **138 metres (about 453 feet)** above sea level.","tool_calls":1,"judge":true,"stopped":false}
```

# Notes
Framework: FastAPI
Route: /ask
Health check works: yes
Example question and reply: See above in `Responses` section
Judge included: yes

What I still find shaky, in full sentences:
- Nothing for today.