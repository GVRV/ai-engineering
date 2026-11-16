from fastapi import FastAPI
from pydantic import BaseModel
from eval_loop import ask_model, process_data
from eval_tools import get_tool_calls

app = FastAPI()


@app.get("/health")
def read_root():
    return {"ok": True}


class AskRequest(BaseModel):
    question: str


class Answer(BaseModel):
    reply: str
    tool_calls: list[str]
    judge: bool
    stopped: bool


@app.post("/ask")
def read_item(ask: AskRequest) -> Answer:
    response = dict()
    reply = ask_model(ask.question)
    process_data(response, reply)
    response["tool_calls"] = get_tool_calls(reply["paste"])
    return Answer(
        reply=response["reply"],
        tool_calls=response["tool_calls"],
        judge=response["judge"],
        stopped=response["stopped"],
    )