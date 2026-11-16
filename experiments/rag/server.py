from fastapi import FastAPI
from pydantic import BaseModel
from eval_loop import ask_model, process_data

app = FastAPI()


@app.get("/health")
def read_root():
    return {"ok": True}


class AskRequest(BaseModel):
    question: str


class Answer(BaseModel):
    reply: str
    tool_calls: int
    judge: bool
    stopped: bool


@app.post("/ask")
def read_item(ask: AskRequest) -> Answer:
    response = dict()
    reply = ask_model(ask.question)
    process_data(response, reply)
    return Answer(
        reply=response["reply"],
        tool_calls=response["calls"],
        judge=response["judge"],
        stopped=response["stopped"],
    )