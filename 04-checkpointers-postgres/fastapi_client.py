"""A client for `fastapi_app.py`.

Create a thread, run two turns on it, read the state back -- the same
two-turn conversation `checkpointer_basics.py` runs, just over HTTP
instead of in-process.

Run `fastapi_app.py` first (see the README), then run this script.
"""

import httpx

BASE_URL = "http://localhost:8000"

if __name__ == "__main__":
    with httpx.Client(base_url=BASE_URL) as client:
        thread = client.post("/threads").json()
        thread_id = thread["thread_id"]
        print(f"Created thread {thread_id}\n")

        print("=== Turn 1 ===")
        result = client.post(
            f"/threads/{thread_id}/runs",
            json={"message": "What's 6 times 7?"},
        ).json()
        print(f"-> {result['reply']}")

        print("\n=== Turn 2 (same thread_id: remembers turn 1) ===")
        result = client.post(
            f"/threads/{thread_id}/runs",
            json={"message": "Now add 8 to that."},
        ).json()
        print(f"-> {result['reply']}")

        state = client.get(f"/threads/{thread_id}/state").json()
        print(f"\n{state['message_count']} messages total on thread {thread_id}.")
