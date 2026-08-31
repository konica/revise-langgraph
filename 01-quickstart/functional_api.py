"""LangGraph quickstart: the same calculator agent built with the Functional API.

Instead of wiring nodes and edges explicitly, the control flow (call the LLM,
run any tools it asked for, loop until it stops asking) is just a Python
`while` loop inside a single `@entrypoint`-decorated function. `@task` marks
the steps LangGraph should track and checkpoint individually.

Reference: https://docs.langchain.com/oss/python/langgraph/quickstart
"""

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langgraph.func import entrypoint, task
from langgraph.graph.message import add_messages

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

model = init_chat_model("claude-sonnet-4-6", temperature=0)
model_with_tools = model.bind_tools(tools)


# --- Tasks ---------------------------------------------------------------

@task
def call_llm(messages: list[AnyMessage]):
    """Ask the LLM what to do next: call a tool or answer directly."""
    return model_with_tools.invoke(
        [
            SystemMessage(
                content="You are a helpful assistant tasked with performing "
                "arithmetic on a set of inputs."
            )
        ]
        + messages
    )


@task
def call_tool(tool_call: dict):
    """Execute a single tool call the LLM requested."""
    current_tool = tools_by_name[tool_call["name"]]
    return current_tool.invoke(tool_call)


# --- Entrypoint: the agent's control flow ---------------------------------

@entrypoint()
def agent(messages: list[AnyMessage]):
    model_response = call_llm(messages).result()

    while model_response.tool_calls:
        tool_result_futures = [call_tool(tc) for tc in model_response.tool_calls]
        tool_results = [fut.result() for fut in tool_result_futures]
        messages = add_messages(messages, [model_response, *tool_results])
        model_response = call_llm(messages).result()

    return add_messages(messages, [model_response])


if __name__ == "__main__":
    result = agent.invoke([HumanMessage(content="Add 3 and 4.")])
    for m in result:
        m.pretty_print()
