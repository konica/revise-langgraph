"""Case study: a customer support email agent.

Walks through every idea in "Thinking in LangGraph" using one graph:

- State design (Step 3): raw data only, formatted into prompts on-demand.
- Nodes as functions (Step 4) that return `Command` objects to update state
  *and* choose the next node in a single return value — routing lives
  inside nodes instead of in a web of conditional edges.
- All four error-handling strategies from the doc's table, each attached to
  the node where it naturally belongs:
    * transient      -> `search_documentation` has a `RetryPolicy`
    * LLM-recoverable -> `bug_tracking` stores a failure message in state
                         instead of crashing, so `draft_response` can see
                         and react to it
    * user-fixable    -> `human_review` calls `interrupt()` and pauses
    * unexpected      -> `send_reply` lets a simulated failure bubble up
- Human-in-the-loop with `interrupt()` + a checkpointer, resumed via
  `Command(resume=...)`.

Reference: https://docs.langchain.com/oss/python/langgraph/thinking-in-langgraph
"""

from typing import Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, RetryPolicy, interrupt
from typing_extensions import TypedDict

load_dotenv()


# --- State -------------------------------------------------------------
#
# Only raw data lives here: the email itself, the LLM's classification
# dict, raw search hits, and the draft. Nothing pre-formatted as a prompt
# string — each node formats what it needs when it needs it (Step 3).

class EmailClassification(TypedDict):
    intent: Literal["question", "bug", "billing", "feature", "complex"]
    urgency: Literal["low", "medium", "high", "critical"]
    topic: str
    summary: str


class EmailAgentState(TypedDict):
    email_content: str
    sender_email: str
    email_id: str

    classification: EmailClassification | None
    search_results: list[str] | None
    draft_response: str | None

    # Only used by the "unexpected error" demo at the bottom of this file.
    _simulate_send_failure: bool | None


# --- Model ---------------------------------------------------------------

model = init_chat_model("claude-haiku-4-5", temperature=0)
classifier = model.with_structured_output(EmailClassification)


# --- Simulated external services ------------------------------------------
# Stand-ins for a knowledge base, a bug tracker, and an email service so
# the case study runs offline and deterministically. Only their LLM calls
# (classification, drafting) hit the real API.

class SearchAPIError(Exception):
    pass


class BugTrackerError(Exception):
    pass


class EmailServiceError(Exception):
    pass


_search_attempts = {"count": 0}


def _fake_doc_search(query: str) -> list[str]:
    """Raises once, then succeeds — enough to exercise a real retry."""
    _search_attempts["count"] += 1
    if _search_attempts["count"] == 1:
        raise SearchAPIError("knowledge base timed out")
    return [
        "Reset password via Settings > Security > Change Password.",
        "Password must be at least 12 characters.",
        f"(matched on: {query})",
    ]


def _fake_bug_tracker_create(summary: str) -> str:
    if "crash" in summary.lower():
        raise BugTrackerError("tracker API rejected payload: missing severity field")
    return "BUG-12345"


def _fake_email_send(body: str, *, simulate_failure: bool = False) -> None:
    if simulate_failure:
        raise EmailServiceError("SMTP relay refused connection")
    print(f"    [sent] {body[:90]}{'...' if len(body) > 90 else ''}")


# --- Nodes -----------------------------------------------------------------

def read_email(state: EmailAgentState) -> dict:
    """Data step: no decisions, just make the raw email available in state."""
    return {}


def classify_intent(
    state: EmailAgentState,
) -> Command[Literal["search_documentation", "human_review", "draft_response", "bug_tracking"]]:
    """LLM step: classify, then route based on the result.

    Static context (the prompt instructions) and dynamic context (the
    email itself) are combined here, on-demand — state only ever held the
    raw email content and sender.
    """
    prompt = f"""Analyze this customer support email and classify it.

Email: {state['email_content']}
From: {state['sender_email']}

Provide the intent, urgency, a short topic, and a one-sentence summary."""

    classification = classifier.invoke(prompt)

    if classification["intent"] == "billing" or classification["urgency"] == "critical":
        # Critical/billing issues skip the AI draft entirely — a human
        # writes the reply themselves in `human_review`.
        goto = "human_review"
    elif classification["intent"] in ("question", "feature"):
        goto = "search_documentation"
    elif classification["intent"] == "bug":
        goto = "bug_tracking"
    else:
        goto = "draft_response"

    return Command(update={"classification": classification}, goto=goto)


def search_documentation(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Data step with a transient-error strategy: retry_policy on this
    node (set below, at add_node time) automatically re-runs the whole
    function if it raises — no try/except needed for that case."""
    classification = state.get("classification") or {}
    query = f"{classification.get('intent', '')} {classification.get('topic', '')}"
    search_results = _fake_doc_search(query)  # may raise SearchAPIError -> retried
    return Command(update={"search_results": search_results}, goto="draft_response")


def bug_tracking(state: EmailAgentState) -> Command[Literal["draft_response"]]:
    """Action step with an LLM-recoverable strategy: on failure, store the
    problem in state instead of crashing, so the next LLM step
    (draft_response) can see it and write around it."""
    classification = state.get("classification") or {}
    try:
        ticket_id = _fake_bug_tracker_create(classification.get("summary", ""))
        note = f"Bug ticket {ticket_id} created."
    except BugTrackerError as e:
        note = f"Could not auto-file a ticket ({e}); flagged for manual triage."

    return Command(update={"search_results": [note]}, goto="draft_response")


def draft_response(state: EmailAgentState) -> Command[Literal["human_review", "send_reply"]]:
    """LLM step: generate a reply, formatting state's raw data into a
    prompt only now, then decide whether it needs human review."""
    classification = state.get("classification") or {}

    context = ""
    if state.get("search_results"):
        context = "Relevant context:\n" + "\n".join(f"- {r}" for r in state["search_results"])

    prompt = f"""Draft a short, professional reply to this customer email.

Email: {state['email_content']}
Intent: {classification.get('intent', 'unknown')}
Urgency: {classification.get('urgency', 'medium')}

{context}

Keep it to 2-3 sentences and address their specific concern."""

    response = model.invoke(prompt)

    needs_review = (
        classification.get("urgency") in ("high", "critical")
        or classification.get("intent") == "complex"
    )
    goto = "human_review" if needs_review else "send_reply"

    return Command(update={"draft_response": response.content}, goto=goto)


def human_review(state: EmailAgentState) -> Command[Literal["send_reply", "__end__"]]:
    """User-fixable strategy: interrupt() pauses the graph, persists state
    via the checkpointer, and waits — potentially indefinitely — for a
    human decision. `interrupt()` must be the first thing in the node;
    everything before it re-runs when the node resumes."""
    classification = state.get("classification") or {}

    decision = interrupt(
        {
            "email_id": state.get("email_id", ""),
            "original_email": state.get("email_content", ""),
            "draft_response": state.get("draft_response", ""),
            "urgency": classification.get("urgency"),
            "intent": classification.get("intent"),
            "action": "Approve, edit, or reject this response.",
        }
    )

    if decision.get("approved"):
        return Command(
            update={"draft_response": decision.get("edited_response", state.get("draft_response", ""))},
            goto="send_reply",
        )
    return Command(update={}, goto=END)


def send_reply(state: EmailAgentState) -> dict:
    """Action step, unexpected-error strategy: don't catch what you can't
    handle. A simulated send failure is allowed to bubble up as a real
    exception rather than being swallowed."""
    _fake_email_send(state["draft_response"] or "", simulate_failure=state.get("_simulate_send_failure", False))
    return {}


# --- Wire it together (Step 5) ---------------------------------------------
#
# Only the edges that are always the same are declared here. Every
# decision (which node runs next) lives inside the node that made the
# decision, via the Command objects above.

builder = StateGraph(EmailAgentState)
builder.add_node("read_email", read_email)
builder.add_node("classify_intent", classify_intent)
builder.add_node(
    "search_documentation",
    search_documentation,
    retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.1),
)
builder.add_node("bug_tracking", bug_tracking)
builder.add_node("draft_response", draft_response)
builder.add_node("human_review", human_review)
builder.add_node("send_reply", send_reply)

builder.add_edge(START, "read_email")
builder.add_edge("read_email", "classify_intent")
builder.add_edge("send_reply", END)

checkpointer = InMemorySaver()
agent = builder.compile(checkpointer=checkpointer)


# --- Try it out --------------------------------------------------------

SCENARIOS = [
    ("simple-question", "How do I reset my password?", "user1@example.com"),
    ("bug-report", "The export feature crashes when I select PDF format.", "user2@example.com"),
    ("urgent-billing", "I was charged twice for my subscription! This is urgent!", "user3@example.com"),
    ("feature-request", "Can you add dark mode to the mobile app?", "user4@example.com"),
    ("complex-technical", "Our API integration fails intermittently with 504 errors.", "user5@example.com"),
]


def run_scenario(thread_id: str, email_content: str, sender_email: str) -> None:
    print(f"\n=== {thread_id} ===")
    print(f"  email: {email_content!r}")

    config = {"configurable": {"thread_id": thread_id}}
    initial_state: EmailAgentState = {
        "email_content": email_content,
        "sender_email": sender_email,
        "email_id": thread_id,
        "classification": None,
        "search_results": None,
        "draft_response": None,
        "_simulate_send_failure": False,
    }

    result = agent.invoke(initial_state, config)

    if interrupts := result.get("__interrupt__"):
        payload = interrupts[0].value
        print(f"  -> paused for human review: urgency={payload['urgency']} intent={payload['intent']}")
        print(f"     draft shown to reviewer: {payload['draft_response']!r}")

        # Simulate a human approving (optionally editing) the response.
        human_decision = {
            "approved": True,
            "edited_response": payload["draft_response"]
            or "Thanks for reaching out — I've personally taken over your case and issued a refund.",
        }
        result = agent.invoke(Command(resume=human_decision), config)

    classification = result.get("classification") or {}
    print(f"  classification: intent={classification.get('intent')} urgency={classification.get('urgency')}")
    print(f"  final reply: {result.get('draft_response')!r}")


def run_unexpected_error_demo() -> None:
    """Shows the fourth error strategy: unexpected errors are not caught
    anywhere in `send_reply`, so they propagate out of `agent.invoke()`
    as a real exception instead of being silently swallowed."""
    print("\n=== unexpected-error-demo ===")
    config = {"configurable": {"thread_id": "unexpected-error-demo"}}
    initial_state: EmailAgentState = {
        "email_content": "Quick question about billing cycles.",
        "sender_email": "user6@example.com",
        "email_id": "unexpected-error-demo",
        "classification": None,
        "search_results": None,
        "draft_response": None,
        "_simulate_send_failure": True,
    }
    try:
        agent.invoke(initial_state, config)
    except EmailServiceError as e:
        print(f"  -> unexpected error bubbled up as expected: {e!r}")


if __name__ == "__main__":
    for thread_id, email_content, sender_email in SCENARIOS:
        run_scenario(thread_id, email_content, sender_email)
    run_unexpected_error_demo()
