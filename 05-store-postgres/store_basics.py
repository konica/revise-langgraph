"""The calculator agent with long-term memory: PostgresStore + PostgresSaver.

`04-checkpointers-postgres` showed the checkpointer keeping one
conversation's state alive across process restarts. This script adds the
other half of LangGraph persistence: a store that remembers things *across*
conversations, keyed by `user_id` instead of `thread_id`.

The two personas below make the difference concrete:

- **Alice** has a conversation, starts a brand-new thread (empty
  checkpointer state -- no message history at all), and asks about her
  last calculation anyway. She gets an answer, because `llm_call` reads
  from the store, not from thread history.
- **Bob** asks the same question on his own first-ever thread and draws a
  blank -- the store is namespaced per `user_id`, so Alice's calculations
  were never visible to him in the first place.

Reference: https://docs.langchain.com/oss/python/langgraph/stores
"""

import os

from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore

from calculator_agent import Context, MessagesState, agent_builder


def last_answer(result: MessagesState) -> str:
    return result["messages"][-1].text


if __name__ == "__main__":
    db_uri = os.environ["DATABASE_URL"]

    # Both `setup()` calls create their own tables (checkpoints/writes for
    # the saver, a separate store table for the store) in the same
    # database. Nothing about running them together is special -- a real
    # deployment usually points both at the same Postgres instance.
    with PostgresSaver.from_conn_string(db_uri) as checkpointer, \
            PostgresStore.from_conn_string(db_uri) as store:
        checkpointer.setup()
        store.setup()
        agent = agent_builder.compile(checkpointer=checkpointer, store=store)

        alice = Context(user_id="alice")

        print("=== Alice, thread 1, turn 1 ===")
        result = agent.invoke(
            {"messages": [HumanMessage("What's 6 times 7?")]},
            {"configurable": {"thread_id": "alice-thread-1"}},
            context=alice,
        )
        print(f"-> {last_answer(result)}")

        print("\n=== Alice, thread 1, turn 2 ===")
        result = agent.invoke(
            {"messages": [HumanMessage("What's 100 divided by 4?")]},
            {"configurable": {"thread_id": "alice-thread-1"}},
            context=alice,
        )
        print(f"-> {last_answer(result)}")

        print("\n=== What's actually in the store for Alice ===")
        items = store.search(("alice", "calculations"))
        print(f"{len(items)} memories, most recently updated first:")
        for item in items:
            print(f"  {item.value['summary']}  (updated_at={item.updated_at})")

        # --- Cross-thread recall -------------------------------------------
        #
        # A *different* thread_id: the checkpointer has never seen this
        # thread before, so get_state would show no history at all. The
        # store doesn't care -- it's keyed by user_id, and Alice's context
        # still says user_id="alice".
        print("\n=== Alice, thread 2 (brand new thread, same user) ===")
        thread_2_config = {"configurable": {"thread_id": "alice-thread-2"}}
        history_before = list(agent.get_state_history(thread_2_config))
        print(f"checkpointer history for this thread before any run: {len(history_before)} checkpoints")
        result = agent.invoke(
            {"messages": [HumanMessage("What was my most recent calculation?")]},
            thread_2_config,
            context=alice,
        )
        print(f"-> {last_answer(result)}")

        # --- Namespace isolation ---------------------------------------
        #
        # Bob's very first thread. Same question, different user_id --
        # a different namespace in the store, so none of Alice's
        # calculations are visible to him.
        print("\n=== Bob, thread 1 (his first ever thread) ===")
        result = agent.invoke(
            {"messages": [HumanMessage("What was my most recent calculation?")]},
            {"configurable": {"thread_id": "bob-thread-1"}},
            context=Context(user_id="bob"),
        )
        print(f"-> {last_answer(result)}")
