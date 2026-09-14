"""A client for `fastapi_app.py`.

Create a thread, run two turns on it, read the state back -- the same
two-turn conversation `checkpointer_basics.py` runs, just over HTTP
instead of in-process. Then a second thread for the same user, to show
the store recalling a calculation across it.

Run `fastapi_app.py` first (see the README), then run this script.
"""

import httpx

BASE_URL = "http://localhost:8000"
USER_ID = "demo-user"

if __name__ == "__main__":
    with httpx.Client(base_url=BASE_URL) as client:
        thread = client.post("/threads").json()
        thread_id = thread["thread_id"]
        print(f"Created thread {thread_id}\n")

        print("=== Turn 1 ===")
        result = client.post(
            f"/threads/{thread_id}/runs",
            json={"message": "What's 6 times 7?", "user_id": USER_ID},
        ).json()
        print(f"-> {result['reply']}")

        print("\n=== Turn 2 (same thread_id: remembers turn 1) ===")
        result = client.post(
            f"/threads/{thread_id}/runs",
            json={"message": "Now add 8 to that.", "user_id": USER_ID},
        ).json()
        print(f"-> {result['reply']}")

        state = client.get(f"/threads/{thread_id}/state").json()
        print(f"\n{state['message_count']} messages total on thread {thread_id}.")

        calcs = client.get(f"/users/{USER_ID}/calculations").json()
        print(f"\n=== What the store remembers for {USER_ID} ===")
        for c in calcs["calculations"]:
            print(f"  {c}")

        print("\n=== Brand-new thread, same user_id: the store remembers anyway ===")
        thread_2 = client.post("/threads").json()
        thread_2_id = thread_2["thread_id"]
        result = client.post(
            f"/threads/{thread_2_id}/runs",
            json={"message": "What was my most recent calculation?", "user_id": USER_ID},
        ).json()
        print(f"-> {result['reply']}")
