"""Hosts the calculator agent (`calculator_agent.py`) over HTTP, without
Agent Server.

Deploying to your own infrastructure -- an AWS ECS/Fargate service, a
Lambda behind API Gateway, a plain EC2 box, `docker run`, whatever runs a
Python web server -- means there's no Agent Server wiring up a
checkpointer or a Threads API for you. This file is that wiring, written
by hand: it compiles `calculator_agent.agent_builder` with an
`AsyncPostgresSaver` it owns (the same pool pattern as
`production_pool.py`), and implements the two operations a caller needs
itself:

- `POST /threads`           -- mint a new thread_id (a UUID).
- `POST /threads/{id}/runs` -- call `agent.ainvoke()` with that thread_id.

Everything agent-specific (tools, model, graph) lives in
`calculator_agent.py`; this file only knows how to serve it.

Reference: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel

from calculator_agent import agent_builder


# --- App lifecycle: open the pool + checkpointer once, not per-request -----

@asynccontextmanager
async def lifespan(app: FastAPI):
    db_uri = os.environ["DATABASE_URL"]
    async with AsyncConnectionPool(
        conninfo=db_uri,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    ) as pool:
        await pool.open(wait=True)
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        # Compiled once at startup and reused for every request -- this is
        # the "server" role Agent Server would otherwise play: import the
        # graph, wire a checkpointer, keep it running.
        app.state.agent = agent_builder.compile(checkpointer=checkpointer)
        yield


app = FastAPI(
    title="Calculator agent (no Agent Server)",
    lifespan=lifespan,
)


class CreateThreadResponse(BaseModel):
    thread_id: str


class RunRequest(BaseModel):
    message: str


class RunResponse(BaseModel):
    reply: str
    message_count: int


class ThreadStateResponse(BaseModel):
    thread_id: str
    message_count: int
    next: list[str]


@app.post("/threads", response_model=CreateThreadResponse)
async def create_thread() -> CreateThreadResponse:
    """Start a new conversation. Nothing talks to Postgres here -- a
    thread only really exists once the checkpointer writes its first
    checkpoint, which happens on the first run below. The UUID just has
    to be unique; the checkpointer doesn't need to be told about it in
    advance."""
    return CreateThreadResponse(thread_id=str(uuid.uuid4()))


@app.post("/threads/{thread_id}/runs", response_model=RunResponse)
async def run_thread(thread_id: str, body: RunRequest) -> RunResponse:
    config = {"configurable": {"thread_id": thread_id}}
    result = await app.state.agent.ainvoke(
        {"messages": [HumanMessage(body.message)]}, config
    )
    return RunResponse(
        reply=result["messages"][-1].text,
        message_count=len(result["messages"]),
    )


@app.get("/threads/{thread_id}/state", response_model=ThreadStateResponse)
async def get_thread_state(thread_id: str) -> ThreadStateResponse:
    config = {"configurable": {"thread_id": thread_id}}
    snapshot = await app.state.agent.aget_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail="Unknown thread_id")
    return ThreadStateResponse(
        thread_id=thread_id,
        message_count=len(snapshot.values.get("messages", [])),
        next=list(snapshot.next),
    )


@app.get("/health")
async def health() -> dict:
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
