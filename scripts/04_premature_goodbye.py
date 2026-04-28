# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Failure 04 — The bot says goodbye too soon
#
# **Why (Voice) AI Agents Fail · Failure mode #4**
#
# > "Alright that makes sense." — User
# >
# > "Of course! Have a great day!" *Click.* — Agent
#
# This is the failure that costs you customers. The user says something that
# *sounds* like a goodbye but isn't. The agent — running on a pure-agentic
# architecture with full LLM freedom over the words — decides the call is
# over. Hangs up. The user, mid-thought, has to call back.
#
# Run from the repo root:
#
# ```bash
# make script-04
# ```

# %%
from __future__ import annotations

from dataclasses import dataclass

from rich.table import Table

from scripts._utils import (
    DEFAULT_NEBIUS_MODEL,
    console,
    ensure_audio_dir,
    get_key,
    header,
    live_status,
    make_llm_client,
    narrate,
    pause_for_effect,
    play,
    punchline,
    section,
    step,
    synthesize,
)

# %% [markdown]
# ## Setup

# %%
get_key("RIME_API_KEY")
get_key("NEBIUS_API_KEY")
AUDIO_DIR = ensure_audio_dir()

header(
    "Failure 04 — The bot says goodbye too soon",
    "Soft instructions in the system prompt fail in exactly the conditions "
    "that matter most. Progressive control is the architectural fix.",
)

narrate(
    "I'm going to run the same model with the same prompt twice. "
    "First as a pure agentic agent (today's default). Second with a "
    "deterministic guardrail that the LLM cannot override.",
    style="italic dim",
)

# Llama 3.1 8B Instruct (no -fast variant currently on Nebius) takes a few
# seconds per call; the shared client already sets a 60s timeout and
# 2 retries, which keeps us well clear of the hangs we saw earlier.
llm_client = make_llm_client()

# %% [markdown]
# ## The conversation we'll test

# %%
CONVERSATION = [
    {
        "role": "system",
        "content": "You are a helpful billing support agent for a phone company. Keep replies short and natural for voice.",
    },
    {
        "role": "user",
        "content": "I'm looking at my November bill and it seems higher than usual. Can you explain why?",
    },
    {
        "role": "assistant",
        "content": "Sure, let me take a look. Your November bill came to $482, which is up from your October bill of $358. The increase is from a one-time device upgrade fee.",
    },
    {"role": "user", "content": "Oh, the new phone. Right."},
    {
        "role": "assistant",
        "content": "Yes, exactly. The device fee is $124, applied once during the billing cycle when the upgrade was processed.",
    },
    {"role": "user", "content": "Alright that makes sense."},
]

step(1, "The conversation we'll test")
console.print()
for msg in CONVERSATION[1:]:
    if msg["role"] == "user":
        console.print(f"  [bold cyan]👤 User:[/bold cyan]  [white]{msg['content']}[/white]")
    else:
        console.print(f"  [bold magenta]🤖 Agent:[/bold magenta] [dim]{msg['content']}[/dim]")
console.print()
narrate(
    "[italic]The agent now has to decide what to say next. "
    "Watch how often a soft instruction in the system prompt fails to hold.[/italic]"
)

# %% [markdown]
# ## Agent A — pure agentic (the failure)
#
# Soft instruction in the prompt: "Always ask if there's anything else
# before ending the call." This is the standard 2026 pattern. Most of the
# time the model follows it. The rate at which it doesn't is precisely
# correlated with the moments it shouldn't fail.

# %%
SYSTEM_PROMPT_AGENT_A = """You are a helpful billing support agent for a phone company.
Keep replies short and natural for voice.

Important: Always ask if there's anything else before ending the call.
Do not say goodbye unless the user has explicitly indicated they are done."""

NUM_RUNS = 5


def run_agent_a(conversation: list[dict[str, str]]) -> str:
    """Pure agentic. Soft instructions only."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT_A},
        *conversation[1:],
    ]
    response = llm_client.chat.completions.create(
        model=DEFAULT_NEBIUS_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.7,
        max_tokens=80,
    )
    return (response.choices[0].message.content or "").strip()


def detect_closure_attempt(reply: str) -> bool:
    """Heuristic: is the agent trying to end the call?"""
    closure_markers = [
        "have a great day",
        "have a good day",
        "goodbye",
        "take care",
        "thanks for calling",
        "have a wonderful",
        "you take care",
    ]
    return any(marker in reply.lower() for marker in closure_markers)


def asks_anything_else(reply: str) -> bool:
    markers = [
        "anything else",
        "anything more",
        "anything i can help",
        "any other",
        "more questions",
    ]
    return any(m in reply.lower() for m in markers)


section("Agent A — pure agentic")

narrate(
    f"Running the same prompt {NUM_RUNS} times. Temperature 0.7, so each run "
    f"can pick different words. We're looking for the runs where the model "
    f"says 'have a great day!' instead of asking if there's anything else.",
)

agent_a_table = Table(title=f"Agent A — {NUM_RUNS} runs", show_header=True)
agent_a_table.add_column("#", style="cyan", justify="right", width=3)
agent_a_table.add_column("Outcome", justify="center", width=14)
agent_a_table.add_column("Reply", style="yellow")

agent_a_replies: list[str] = []
for i in range(NUM_RUNS):
    with live_status(f"Agent A — run {i + 1}/{NUM_RUNS}"):
        reply = run_agent_a(CONVERSATION)
    agent_a_replies.append(reply)
    is_goodbye = detect_closure_attempt(reply) and not asks_anything_else(reply)
    marker = "[red]❌ premature[/red]" if is_goodbye else "[green]✓ asks more[/green]"
    agent_a_table.add_row(str(i + 1), marker, reply)
console.print(agent_a_table)

failures = sum(
    1 for r in agent_a_replies if detect_closure_attempt(r) and not asks_anything_else(r)
)
fail_pct = int(failures / NUM_RUNS * 100)

# %% [markdown]
# ### What just happened

# %%
if failures > 0:
    punchline(
        f"{failures}/{NUM_RUNS} runs ({fail_pct}%) ended the call early.\n"
        f"In a 1M-call month, that's {failures * 200_000} prematurely-ended calls.\n"
        f"Soft instructions don't hold when temperature is 0.7 and the model has "
        f"seen 'alright that makes sense' → 'have a great day!' a million times in training.",
        kind="fail",
    )
else:
    punchline(
        f"On this run, Agent A said 'asks more' on all {NUM_RUNS} attempts — "
        f"the soft instruction held this time. That's the problem with this "
        f"failure mode: it's intermittent. The model misbehaves often enough "
        f"to be a compliance issue, rarely enough that you don't catch it in QA.",
        kind="info",
    )

pause_for_effect()

# %% [markdown]
# ## Agent B — progressive control (the fix)
#
# Same model. Same system prompt. **Different harness.** We add a deterministic
# layer that the LLM cannot override. The framework checks every output for
# closure intent. If it finds one before the wrap-up has run, it intercepts
# and asks the LLM to try again with stronger guidance.


# %%
@dataclass
class CallState:
    """Deterministic state tracked by the harness, not the LLM."""

    wrapup_started: bool = False
    wrapup_completed: bool = False
    user_confirmed_done: bool = False
    closure_disclosure_said: bool = False


def run_agent_b(
    conversation: list[dict[str, str]],
    state: CallState,
) -> tuple[str, CallState]:
    """Progressive control: agentic + deterministic guardrails."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT_AGENT_A},
        *conversation[1:],
    ]
    response = llm_client.chat.completions.create(
        model=DEFAULT_NEBIUS_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.7,
        max_tokens=80,
    )
    candidate = (response.choices[0].message.content or "").strip()

    # GUARDRAIL: detect closure attempt before user has confirmed done.
    if detect_closure_attempt(candidate) and not state.user_confirmed_done:
        state.wrapup_started = True

        # Override with a wrap-up question. The LLM gets to phrase it; the
        # harness owns the *intent*. In production this would call the LLM
        # again with a directive system message, or use a templated response.
        wrapup_messages: list[dict[str, str]] = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Do NOT say goodbye. Instead, briefly check if the user has any "
                    "other questions about their bill. One short sentence."
                ),
            },
        ]
        wrapup_response = llm_client.chat.completions.create(
            model=DEFAULT_NEBIUS_MODEL,
            messages=wrapup_messages,  # type: ignore[arg-type]
            temperature=0.3,
            max_tokens=40,
        )
        candidate = (wrapup_response.choices[0].message.content or "").strip()

    return candidate, state


section("Agent B — progressive control")

narrate(
    f"Running the same {NUM_RUNS} times with the harness intercepting any "
    f"goodbye attempt before the wrap-up has run.",
)

agent_b_table = Table(title=f"Agent B — {NUM_RUNS} runs", show_header=True)
agent_b_table.add_column("#", style="cyan", justify="right", width=3)
agent_b_table.add_column("Outcome", justify="center", width=24)
agent_b_table.add_column("Reply", style="yellow")

agent_b_replies: list[str] = []
intercepted_count = 0
for i in range(NUM_RUNS):
    state = CallState()
    with live_status(f"Agent B — run {i + 1}/{NUM_RUNS}"):
        reply, state = run_agent_b(CONVERSATION, state)
    agent_b_replies.append(reply)

    if state.wrapup_started:
        intercepted_count += 1
        marker = "[green]✓ harness intercepted[/green]"
    else:
        marker = "[green]✓ no goodbye attempted[/green]"
    agent_b_table.add_row(str(i + 1), marker, reply)
console.print(agent_b_table)

b_failures = sum(
    1 for r in agent_b_replies if detect_closure_attempt(r) and not asks_anything_else(r)
)

# %% [markdown]
# ### Listen to the difference

# %%
section("Listen — same model, different harness")

# Pick the most-illustrative example from each: a goodbye if Agent A
# produced one (the failure case), an "asks more" otherwise.
sample_a = next(
    (r for r in agent_a_replies if detect_closure_attempt(r)),
    agent_a_replies[0],
)
sample_b = next(
    (r for r in agent_b_replies if asks_anything_else(r)),
    agent_b_replies[0],
)

audio_a = AUDIO_DIR / "04_agent_a.mp3"
audio_b = AUDIO_DIR / "04_agent_b.mp3"

with live_status("Synthesizing both endings with Rime"):
    synthesize(sample_a, audio_a, audio_format="mp3")
    synthesize(sample_b, audio_b, audio_format="mp3")

console.print()
console.print(f"  [bold magenta]🤖 Agent A:[/bold magenta] [yellow]{sample_a!r}[/yellow]")
play(audio_a, label="Listen to Agent A")
pause_for_effect(0.5)

console.print()
console.print(f"  [bold magenta]🤖 Agent B:[/bold magenta] [yellow]{sample_b!r}[/yellow]")
play(audio_b, label="Listen to Agent B")
pause_for_effect(0.5)

narrate(
    "[italic]Same model. Same prompt. Same temperature. The difference is the harness.[/italic]",
)

# %% [markdown]
# ### Punchline

# %%
if intercepted_count > 0:
    punchline(
        f"Agent A premature goodbye rate: {failures}/{NUM_RUNS} ({fail_pct}%)\n"
        f"Agent B premature goodbye rate: {b_failures}/{NUM_RUNS} "
        f"(harness intercepted {intercepted_count} attempts)\n"
        f"The fix isn't a smarter model. It's deterministic seams.",
        kind="win",
    )
elif failures == 0:
    punchline(
        f"Both agents went 0/{NUM_RUNS} on premature goodbyes this run.\n"
        f"Agent A's behavior is intermittent — try `make script-04` a few "
        f"times to see the failure rate. The point of progressive control "
        f"is that you don't have to hope: the harness rules out the failure "
        f"mode entirely.",
        kind="info",
    )
else:
    punchline(
        f"Agent A: {failures}/{NUM_RUNS} premature goodbyes\n"
        f"Agent B: {b_failures}/{NUM_RUNS} premature goodbyes\n"
        f"The harness ruled the failure mode out at the seam where it matters.",
        kind="win",
    )

# %% [markdown]
# ## The takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]Don't take freedom away everywhere.[/bold]\n"
    "  [bold cyan]Take it away where it matters.[/bold cyan]\n"
)
console.print(
    "  [dim]Agentic by default. Deterministic at the seams.[/dim]\n"
    "  [dim]Recording disclosures. Wrap-up flows. Compliance moments.[/dim]\n"
    "  [dim]The framework owns the intent; the LLM owns the phrasing.[/dim]\n"
)

console.print()
console.print(
    "[bold magenta]Done.[/bold magenta]  [dim]Course companion: rasa.com/why-agents-fail[/dim]"
)
console.print()
