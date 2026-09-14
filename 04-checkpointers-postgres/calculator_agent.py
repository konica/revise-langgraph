"""The calculator agent's graph: tools, model, nodes, edges, and memory.

Deliberately has no `compile()` call and no checkpointer/store -- building
the graph is a separate concern from deciding how (or whether) to persist
it and how to serve it. Every script in this scenario compiles the same
`agent_builder` with its own checkpointer (`checkpointer_basics.py`: sync
`PostgresSaver`; `production_pool.py` and `fastapi_app.py`: async
`AsyncPostgresSaver`) and store (`PostgresStore` / `AsyncPostgresStore`).

The nodes below are plain sync functions on purpose, even though two of
those three callers are async. LangGraph runs sync node functions in a
thread pool when you call `.ainvoke()`, so the same `agent_builder` works
with both `.invoke()` and `.ainvoke()` unchanged -- the reverse isn't
true: a graph built from `async def` nodes raises `TypeError: No
synchronous function provided` the moment something calls `.invoke()` on
it.

`tool_node` and `llm_call` also take a `runtime: Runtime[Context]`
parameter, giving them `runtime.store` (long-term, cross-thread memory)
and `runtime.context.user_id` (whose memory). A checkpointer scopes state
to one conversation (`thread_id`); a store scopes it to one user, across
every conversation they ever start -- see the "Why no checkpointer here"
docstring below for how `thread_id` and `user_id` differ.

References:
- https://docs.langchain.com/oss/python/langgraph/quickstart
- https://docs.langchain.com/oss/python/langgraph/stores
"""

import operator
import uuid
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from typing_extensions import Annotated, TypedDict

load_dotenv()


# --- Tools and model ---------------------------------------------------

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


# --- State and context -----------------------------------------------------

class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]
    llm_calls: int


@dataclass
class Context:
    """Runtime context: whose long-term memory to read and write.

    `thread_id` (in `config`) and `user_id` (here) answer different
    questions. `thread_id` scopes *this conversation*'s message history --
    the checkpointer's job. `user_id` scopes memories that should survive
    past this conversation -- the store's job. The same `user_id` across
    different `thread_id`s is what makes cross-thread recall possible: a
    brand new thread still has access to everything stored under that
    user, even though it has no message history of its own.
    """

    user_id: str


MEMORY_NAMESPACE = "calculations"


# --- Nodes -----------------------------------------------------------------

def llm_call(state: MessagesState, runtime: Runtime[Context]):
    """Ask the LLM what to do next, with the user's past calculations
    (from every thread, not just this one) folded into the system prompt."""
    namespace = (runtime.context.user_id, MEMORY_NAMESPACE)
    memories = runtime.store.search(namespace, limit=5)
    if memories:
        recalled = "\n".join(f"- {item.value['summary']}" for item in memories)
    else:
        recalled = "(none yet)"

    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful assistant tasked with "
                        "performing arithmetic on a set of inputs.\n\n"
                        "This user's past calculations, from any prior "
                        "conversation, most recent first:\n"
                        f"{recalled}"
                    )
                ]
                + state["messages"]
            )
        ],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


def tool_node(state: MessagesState, runtime: Runtime[Context]):
    """Execute every tool call the LLM requested, and remember each one in
    the store -- deterministically, with no extra LLM call needed, since
    the tool call already tells us exactly what happened."""
    namespace = (runtime.context.user_id, MEMORY_NAMESPACE)
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        current_tool = tools_by_name[tool_call["name"]]
        observation = current_tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))

        args = ", ".join(f"{k}={v}" for k, v in tool_call["args"].items())
        summary = f"{tool_call['name']}({args}) = {observation}"
        runtime.store.put(namespace, str(uuid.uuid4()), {"summary": summary})

    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Loop back to the tool node while the LLM keeps requesting tools."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# --- Build the graph (not compiled -- callers decide checkpointer/store) ---

agent_builder = StateGraph(MessagesState, context_schema=Context)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
