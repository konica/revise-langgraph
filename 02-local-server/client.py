"""A client for the agent running under Agent Server (`langgraph dev`).

This process never imports `agent.py` or LangGraph itself — it only speaks
HTTP to whatever `langgraph dev` is serving on `http://localhost:2024`. Any
client that can make HTTP requests (curl, Studio, another service) could
stand in for this script.

Run `langgraph dev` from this directory first, then run this script.

Reference: https://docs.langchain.com/oss/python/langgraph/local-server
"""

from langgraph_sdk import get_sync_client

client = get_sync_client(url="http://localhost:2024")

if __name__ == "__main__":
    for chunk in client.runs.stream(
        None,  # threadless run
        "agent",  # the graph name declared in langgraph.json
        input={"messages": [{"role": "human", "content": "Add 3 and 4."}]},
        stream_mode="updates",
    ):
        print(chunk.event)
        print(chunk.data)
