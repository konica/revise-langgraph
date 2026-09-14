"""The calculator agent from `calculator_agent.py`, checkpointed to Postgres.

Swapping `InMemorySaver` for `PostgresSaver` is the whole difference between
a demo and something you could actually put in production: conversation
state survives process restarts because it lives in the database, not in
the Python process's memory.

This script walks through the checkpointer concepts from the docs, in
order: threads, checkpoints, `get_state`, `get_state_history`, and forking
history with `update_state` (time travel).

Reference: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

import os

from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import START

from calculator_agent import MessagesState, agent_builder


def last_answer(result: MessagesState) -> str:
    return result["messages"][-1].text


if __name__ == "__main__":
    db_uri = os.environ["DATABASE_URL"]

    # `setup()` creates the checkpoints/writes tables. It's idempotent, so
    # it's safe to call every run, but in a real deployment you'd run it
    # once as a migration step rather than on every app startup.
    with PostgresSaver.from_conn_string(db_uri) as checkpointer:
        checkpointer.setup()
        agent = agent_builder.compile(checkpointer=checkpointer)

        # Every checkpointed run needs a thread_id: it's the key the
        # checkpointer uses to find this conversation's saved state again.
        config = {"configurable": {"thread_id": "demo-thread-1"}}

        print("=== Turn 1 ===")
        result = agent.invoke(
            {"messages": [HumanMessage("What's 6 times 7?")]}, config
        )
        print(f"-> {last_answer(result)}")

        # Snapshot the state right after turn 1. We'll fork from this point
        # later to show that update_state doesn't touch what actually
        # happened -- it creates a new, separate branch of history.
        after_turn1 = agent.get_state(config)

        print("\n=== Turn 2 (same thread_id, so it remembers turn 1) ===")
        result = agent.invoke(
            {"messages": [HumanMessage("Now add 8 to that.")]}, config
        )
        print(f"-> {last_answer(result)}")

        print("\n=== get_state: latest checkpoint for this thread ===")
        snapshot = agent.get_state(config)
        print(f"messages so far : {len(snapshot.values['messages'])}")
        print(f"next to run     : {snapshot.next}")
        print(f"checkpoint_id   : {snapshot.config['configurable']['checkpoint_id']}")

        print("\n=== get_state_history: every checkpoint saved for this thread ===")
        history = list(agent.get_state_history(config))
        print(f"{len(history)} checkpoints total (one per super-step, newest first)")
        for s in history:
            print(f"  step={s.metadata['step']:>3}  source={s.metadata['source']:<6}  next={s.next}")

        # --- Time travel: fork the conversation from after turn 1 ----------
        #
        # `update_state` writes a *new* checkpoint whose parent is the
        # checkpoint we pass in -- it never mutates the original. Passing
        # `after_turn1.config` (captured before turn 2 ever happened) forks
        # the thread: turn 2's "add 8" branch stays in history untouched,
        # and this new branch becomes the thread's current tip.
        #
        # `as_node=START` tells LangGraph to treat this update the same way
        # a fresh `invoke()` call treats new input: enter at START and take
        # its unconditional edge to `llm_call` next. Without it, the update
        # is attributed to whichever node wrote the last checkpoint
        # (`llm_call`), so the graph would try to run `llm_call`'s
        # *outgoing* conditional edge against our new human message instead.
        print("\n=== Fork from after Turn 1: ask a different follow-up ===")
        fork_config = agent.update_state(
            after_turn1.config,
            {"messages": [HumanMessage("Now divide that by 6.")]},
            as_node=START,
        )
        # Resuming with `None` as input replays from the checkpoint in
        # `fork_config` instead of starting a new run.
        result = agent.invoke(None, fork_config)
        print(f"-> {last_answer(result)}")

        print("\n=== get_state after the fork: the fork is now the thread's tip ===")
        snapshot = agent.get_state(config)
        print(f"messages so far : {len(snapshot.values['messages'])}")
        print(
            "Turn 2's answer is still on disk -- it's just no longer the "
            "branch `thread_id=demo-thread` points to by default. Restart "
            "this script (same DATABASE_URL) and Turn 1's answer is still "
            "there: that's the whole point of a Postgres-backed checkpointer."
        )
