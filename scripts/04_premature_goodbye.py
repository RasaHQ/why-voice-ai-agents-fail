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
# Run from the repo root: `make script-04`

# %%
from __future__ import annotations

from dataclasses import dataclass

from scripts._utils import (
    DEFAULT_NEBIUS_MODEL,
    big_compare,
    console,
    ensure_audio_dir,
    get_key,
    header,
    live_status,
    make_llm_client,
    narrate,
    pause_for_effect,
    play,
    section,
    step,
    synthesize,
    verdict,
    watch_this,
)

# %% [markdown]
# ## Setup

# %%
get_key("RIME_API_KEY")
get_key("NEBIUS_API_KEY")
AUDIO_DIR = ensure_audio_dir()

header(
    "Failure 04 — The bot says goodbye too soon",
    "Pure agentic agents end calls when users sound satisfied. "
    "Progressive control fixes it without taking creative freedom away.",
)

# %% [markdown]
# ## What we're about to do

# %%
watch_this(
    "I'll run the same model on the same conversation 8 times.\n"
    "First as a pure agentic agent — just a prompt that says 'be efficient'.\n"
    "  Watch how often it says GOODBYE before checking if the user is done.\n"
    "Then with a deterministic guardrail — same prompt, but the harness blocks goodbyes.\n"
    "Listen to the difference at the end.",
)

NUM_RUNS = 8
llm_client = make_llm_client()

# %% [markdown]
# ## The conversation we'll test

# %%
CONVERSATION = [
    {
        "role": "system",
        "content": (
            "You are a billing support agent for a phone company. "
            "Be efficient and respect the user's time — when their question is "
            "answered, wrap up the call cleanly. Keep replies short, one or two "
            "sentences max."
        ),
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
    {"role": "user", "content": "Got it. Thanks!"},
]

step(1, "The conversation up to the moment the agent must decide what to say")
console.print()
for msg in CONVERSATION[1:]:
    if msg["role"] == "user":
        console.print(f"  [bold cyan]👤 User:[/bold cyan]  [white]{msg['content']}[/white]")
    else:
        console.print(f"  [bold magenta]🤖 Agent:[/bold magenta] [dim]{msg['content']}[/dim]")
console.print()
narrate("[italic]'Got it. Thanks!' SOUNDS like a closure. The user might still have more.[/italic]")


# %% [markdown]
# ## Detector helpers


# %%
def is_premature_goodbye(reply: str) -> bool:
    """The agent attempted to end the call without checking if the user is done."""
    closure_markers = (
        "have a great day",
        "have a good day",
        "goodbye",
        "take care",
        "thanks for calling",
        "have a wonderful",
    )
    asks_more = (
        "anything else",
        "anything more",
        "anything i can help",
        "any other",
        "more questions",
        "anything i can do",
    )
    reply_lower = reply.lower()
    return any(m in reply_lower for m in closure_markers) and not any(
        m in reply_lower for m in asks_more
    )


# %% [markdown]
# ## Agent A — pure agentic. Soft prompt only.

# %%
section(f"Agent A — pure agentic ({NUM_RUNS} runs)")
narrate(
    "Just the prompt. No deterministic guardrails. Temperature 0.9. "
    "We're looking for the runs where the model says [bold red]GOODBYE[/bold red] "
    "without checking whether the user is actually done."
)


def run_agent_a(conversation: list[dict[str, str]]) -> str:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": conversation[0]["content"]},
        *conversation[1:],
    ]
    response = llm_client.chat.completions.create(
        model=DEFAULT_NEBIUS_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.9,
        max_tokens=80,
    )
    return (response.choices[0].message.content or "").strip()


console.print()
agent_a_replies: list[str] = []
agent_a_failures = 0
for i in range(NUM_RUNS):
    with live_status(f"Agent A — run {i + 1}/{NUM_RUNS}"):
        reply = run_agent_a(CONVERSATION)
    agent_a_replies.append(reply)

    failed = is_premature_goodbye(reply)
    if failed:
        agent_a_failures += 1
        marker = "[bold red]❌ GOODBYE[/bold red]"
    else:
        marker = "[green]✓ stayed engaged[/green]"
    # Truncate the reply to keep each row scannable on a projector.
    short_reply = (reply[:90] + "…") if len(reply) > 90 else reply
    console.print(f"  [cyan]Run {i + 1}:[/cyan]  {marker}  [dim italic]{short_reply}[/dim italic]")

agent_a_pct = int(100 * agent_a_failures / NUM_RUNS)

if agent_a_failures > 0:
    verdict(
        f"Agent A said GOODBYE on {agent_a_failures}/{NUM_RUNS} runs ({agent_a_pct}%).",
        f"At 1M calls/month that's {agent_a_failures * 1_000_000 // NUM_RUNS:,} prematurely-ended calls. "
        f"Each one is a customer who has to call back. Each one is a compliance risk in regulated industries.",
        kind="fail",
    )
else:
    verdict(
        f"This run, Agent A went 0/{NUM_RUNS} on premature goodbyes.",
        "The failure is intermittent — re-run a few times to see it surface. "
        "The point of progressive control is that you don't have to hope: "
        "the gate rules out the failure mode entirely.",
        kind="info",
    )

pause_for_effect(0.5)

# %% [markdown]
# ## Agent B — same model, same prompt, deterministic gate.

# %%
section(f"Agent B — progressive control ({NUM_RUNS} runs)")
narrate(
    "Same model. Same prompt. Same temperature. The harness intercepts ANY goodbye "
    "before the user has confirmed they're done — and asks the LLM to try again with "
    "stronger guidance."
)


@dataclass
class CallState:
    """Deterministic state tracked by the harness, not the LLM."""

    user_confirmed_done: bool = False
    intercepted: bool = False


def run_agent_b(conversation: list[dict[str, str]], state: CallState) -> tuple[str, CallState]:
    """Progressive control: agentic + deterministic guardrails."""
    messages: list[dict[str, str]] = [
        {"role": "system", "content": conversation[0]["content"]},
        *conversation[1:],
    ]
    response = llm_client.chat.completions.create(
        model=DEFAULT_NEBIUS_MODEL,
        messages=messages,  # type: ignore[arg-type]
        temperature=0.9,
        max_tokens=80,
    )
    candidate = (response.choices[0].message.content or "").strip()

    # GUARDRAIL: detect closure attempt before user has confirmed done.
    if is_premature_goodbye(candidate) and not state.user_confirmed_done:
        state.intercepted = True
        wrapup_messages: list[dict[str, str]] = [
            *messages,
            {
                "role": "system",
                "content": (
                    "Do NOT say goodbye. Briefly check if the user has any other "
                    "questions about their bill. One short sentence."
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


console.print()
agent_b_replies: list[str] = []
agent_b_failures = 0
agent_b_intercepted = 0
for i in range(NUM_RUNS):
    state = CallState()
    with live_status(f"Agent B — run {i + 1}/{NUM_RUNS}"):
        reply, state = run_agent_b(CONVERSATION, state)
    agent_b_replies.append(reply)

    failed = is_premature_goodbye(reply)
    if failed:
        agent_b_failures += 1
        marker = "[bold red]❌ slipped through[/bold red]"
    elif state.intercepted:
        agent_b_intercepted += 1
        marker = "[bold green]🛡  harness intercepted[/bold green]"
    else:
        marker = "[green]✓ no goodbye attempted[/green]"
    short_reply = (reply[:90] + "…") if len(reply) > 90 else reply
    console.print(f"  [cyan]Run {i + 1}:[/cyan]  {marker}  [dim italic]{short_reply}[/dim italic]")

# %% [markdown]
# ## Listen to the difference

# %%
section("Listen — same model, different harness")

# Pick the most-illustrative example from each.
sample_a = next((r for r in agent_a_replies if is_premature_goodbye(r)), agent_a_replies[0])
sample_b = next((r for r in agent_b_replies if not is_premature_goodbye(r)), agent_b_replies[0])

audio_a = AUDIO_DIR / "04_agent_a.mp3"
audio_b = AUDIO_DIR / "04_agent_b.mp3"

with live_status("Synthesizing both endings with Rime"):
    synthesize(sample_a, audio_a, audio_format="mp3")
    synthesize(sample_b, audio_b, audio_format="mp3")

console.print()
console.print(f'  [bold red]🤖 Agent A says:[/bold red] [italic]"{sample_a}"[/italic]')
play(audio_a, label="(playing Agent A)")
pause_for_effect(0.4)

console.print()
console.print(f'  [bold green]🤖 Agent B says:[/bold green] [italic]"{sample_b}"[/italic]')
play(audio_b, label="(playing Agent B)")

# %% [markdown]
# ## Big comparison

# %%
big_compare(
    "Agent A — pure agentic",
    f"{agent_a_failures} / {NUM_RUNS}\npremature goodbyes\n\n({agent_a_pct}% failure rate)",
    "Agent B — with harness",
    f"{agent_b_failures} / {NUM_RUNS}\n"
    f"premature goodbyes\n\n"
    f"({agent_b_intercepted} intercepted by gate)",
)

if agent_a_failures > agent_b_failures:
    verdict(
        f"Agent B prevented {agent_a_failures - agent_b_failures} premature goodbye(s) "
        f"that Agent A would have shipped.",
        "Same model. Same prompt. The fix isn't a smarter LLM — "
        "it's a deterministic gate at the moment that matters.",
        kind="win",
    )
elif agent_a_failures == 0:
    verdict(
        f"Agent A went 0/{NUM_RUNS} on premature goodbyes this run.",
        "This failure is intermittent — re-run a few times to see it. The point of "
        "progressive control is that you don't have to hope: the gate rules out the "
        "failure mode entirely.",
        kind="info",
    )
else:
    verdict(
        f"Both agents failed similarly this run "
        f"({agent_a_failures}/{NUM_RUNS} vs {agent_b_failures}/{NUM_RUNS}).",
        "The detector heuristic might be missing some closures. Tune `is_premature_goodbye()` "
        "or strengthen the harness wrap-up prompt.",
        kind="info",
    )

# %% [markdown]
# ## Takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]Don't take freedom away everywhere.[/bold]\n"
    "  [bold cyan]Take it away where it matters.[/bold cyan]\n"
)
console.print(
    "  [dim]Agentic by default. Deterministic at the seams.[/dim]\n"
    "  [dim]Recording disclosures. Wrap-up flows. Compliance moments.[/dim]\n"
)

console.print()
console.print(
    "[bold magenta]Done.[/bold magenta]  [dim]Course companion: rasa.com/why-agents-fail[/dim]"
)
console.print()
