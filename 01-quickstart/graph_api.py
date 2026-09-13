"""LangGraph quickstart: a calculator agent built with the Graph API.

The agent alternates between an LLM node (which decides whether to call a
tool) and a tool node (which executes whatever the LLM asked for), looping
until the LLM stops requesting tools.

Reference: https://docs.langchain.com/oss/python/langgraph/quickstart
"""

import operator
import sys
from typing import Literal

import anthropic
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, SystemMessage, ToolMessage
from langchain.tools import tool
from langgraph.graph import END, START, StateGraph
from typing_extensions import Annotated, TypedDict

load_dotenv()


# --- Tools and model -------------------------------------------------------

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


# max_retries controls how many times the underlying Anthropic SDK client
# retries a 429/5xx with exponential backoff before giving up. The default
# is 2, which isn't enough headroom on a rate-limited (e.g. trial-tier) key.
model = init_chat_model("claude-sonnet-4-6", temperature=0, max_retries=6)
model_with_tools = model.bind_tools(tools)


# --- State -------------------------------------------------------------

class MessagesState(TypedDict):
    messages: Annotated[list, operator.add]
    llm_calls: int


# --- Nodes -------------------------------------------------------------

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


# --- Build and compile the graph ----------------------------------------

agent_builder = StateGraph(MessagesState)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
agent_builder.add_edge("tool_node", "llm_call")

agent = agent_builder.compile()


if __name__ == "__main__":
    try:
        result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")]})
    except anthropic.RateLimitError as e:
        retry_after = e.response.headers.get("retry-after")
        print(
            "Rate limited (429) after exhausting retries."
            + (f" Retry after {retry_after}s." if retry_after else "")
            + " Check your usage tier / limits at "
            "https://console.anthropic.com/settings/limits, or wait and rerun.",
            file=sys.stderr,
        )
        sys.exit(1)

    for m in result["messages"]:
        m.pretty_print()
