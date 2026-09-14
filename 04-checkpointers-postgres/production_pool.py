"""The production shape: AsyncPostgresSaver over a shared connection pool.

`checkpointer_basics.py` opens one connection and walks through it step by
step. A real server doesn't do that -- it serves many threads (many users'
conversations) concurrently through a small pool of reusable connections,
and it awaits the LLM/tool calls instead of blocking on them.

This script serves two users' conversations at the same time, through one
pooled checkpointer, with `asyncio.gather`, and shows their state stays
correctly isolated per `thread_id`.

Reference: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

import asyncio
import operator
import os
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import END, START, StateGraph
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool
from typing_extensions import Annotated, TypedDict

load_dotenv()


# --- Tools and model (same calculator agent as 01-quickstart) --------------

@tool
def multiply(a: int, b: int) -> int:
    """Multiply `a` and `b`."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add `a` and `b`."""
    return a + b


@tool
def divide(a: int, b: int) -> float:
    """Divide `a` and `b`."""
    return a / b


tools = [add, multiply, divide]
tools_by_name = {t.name: t for t in tools}

model = init_chat_model("claude-haiku-4-5", temperature=0)
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]
    llm_calls: int


async def llm_call(state: MessagesState):
    return {
        "messages": [
            await model_with_tools.ainvoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with "
                        "performing arithmetic on a set of inputs."
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


async def tool_node(state: MessagesState):
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        current_tool = tools_by_name[tool_call["name"]]
        observation = await current_tool.ainvoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")


async def handle_turn(agent, user_id: str, text: str) -> None:
    config = {"configurable": {"thread_id": f"user-{user_id}"}}
    result = await agent.ainvoke(
        {"messages": [HumanMessage(text)]},
        config,
        # "sync": don't move on to the next super-step until this one's
        # checkpoint is durably written. Costs a little latency; buys you
        # the guarantee that a crash mid-run can resume instead of losing
        # the turn. See "Durability modes" in the docs.
        durability="sync",
    )
    print(f"[{user_id}] {text!r} -> {result['messages'][-1].text}")


async def main() -> None:
    db_uri = os.environ["DATABASE_URL"]

    # A connection pool, not a single connection: this is what makes the
    # checkpointer safe to share across concurrent requests. The three
    # kwargs below are required by AsyncPostgresSaver -- they're the same
    # ones `from_conn_string` sets for you on a single connection.
    async with AsyncConnectionPool(
        conninfo=db_uri,
        max_size=20,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=False,
    ) as pool:
        await pool.open(wait=True)
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()
        agent = agent_builder.compile(checkpointer=checkpointer)

        # Two different users, two different threads, served concurrently
        # off the same pooled checkpointer -- each thread's state stays
        # isolated because the checkpointer keys everything by thread_id.
        await asyncio.gather(
            handle_turn(agent, "alice", "What's 9 times 9?"),
            handle_turn(agent, "bob", "What's 100 divided by 4?"),
        )

        # Alice's second turn only sees Alice's history, not Bob's.
        await handle_turn(agent, "alice", "Now add 1 to that.")


if __name__ == "__main__":
    asyncio.run(main())
