"""The calculator agent's graph: tools, model, nodes, and edges.

Deliberately has no `compile()` call and no checkpointer -- building the
graph is a separate concern from deciding how (or whether) to persist it
and how to serve it. Every script in this scenario compiles the same
`agent_builder` with a different checkpointer: `checkpointer_basics.py`
(sync `PostgresSaver`), `production_pool.py` and `fastapi_app.py` (async
`AsyncPostgresSaver`).

The nodes below are plain sync functions on purpose, even though two of
those three callers are async. LangGraph runs sync node functions in a
thread pool when you call `.ainvoke()`, so the same `agent_builder` works
with both `.invoke()` and `.ainvoke()` unchanged -- the reverse isn't
true: a graph built from `async def` nodes raises `TypeError: No
synchronous function provided` the moment something calls `.invoke()` on
it.

Reference: https://docs.langchain.com/oss/python/langgraph/quickstart
"""

import operator
from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
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


# --- State ---------------------------------------------------------------

class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]
    llm_calls: int


# --- Nodes -----------------------------------------------------------------

def llm_call(state: MessagesState):
    """Ask the LLM what to do next: call a tool or answer directly."""
    return {
        "messages": [
            model_with_tools.invoke(
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


def tool_node(state: MessagesState):
    """Execute every tool call the LLM requested in its last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        current_tool = tools_by_name[tool_call["name"]]
        observation = current_tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Loop back to the tool node while the LLM keeps requesting tools."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


# --- Build the graph (not compiled -- callers decide the checkpointer) -----

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")
