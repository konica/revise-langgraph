"""The production shape: AsyncPostgresSaver over a shared connection pool.

`checkpointer_basics.py` opens one connection and walks through it step by
step. A real server doesn't do that -- it serves many threads (many users'
conversations) concurrently through a small pool of reusable connections,
awaiting each run instead of blocking on it.

This script serves two users' conversations at the same time, through one
pooled checkpointer, with `asyncio.gather`, and shows their state stays
correctly isolated per `thread_id`.

Reference: https://docs.langchain.com/oss/python/langgraph/checkpointers
"""

import asyncio
import os

from langchain.messages import HumanMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from calculator_agent import agent_builder


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
