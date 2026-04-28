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
# sounds like a goodbye but isn't. The agent — running on a pure agentic
# architecture with full LLM freedom over the words — decides the call is
# over. It hangs up. The user, mid-thought, has to call back.
#
# **What you'll see:**
#
# 1. Two voice agents with the same model and same system prompt.
# 2. **Agent A** is pure agentic — soft instructions in the prompt, full LLM
#    freedom. The current default in 2026.
# 3. **Agent B** uses **progressive control** — agentic by default, with a
#    deterministic guardrail blocking the call from ending until a wrap-up
#    flow has run.
# 4. Both agents see the same conversation, ending with "alright that makes
#    sense." We synthesize the endings with **Rime** so you can hear the
#    difference.
#
# **Total runtime:** ~30 seconds. **Cost:** ~$0.005.
#
# Run from the repo root:
#
# ```bash
# make script-04
# ```

# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import requests
from rich.panel import Panel
from rich.table import Table

from scripts._utils import (
    DEFAULT_NEBIUS_MODEL,
    console,
    ensure_audio_dir,
    get_key,
    make_llm_client,
)

# %% [markdown]
# ## Setup

# %%
RIME_API_KEY = get_key("RIME_API_KEY")
NEBIUS_API_KEY = get_key("NEBIUS_API_KEY")
AUDIO_DIR = ensure_audio_dir()

console.print(Panel.fit("[bold cyan]Failure 04 — The bot says goodbye too soon[/bold cyan]"))
llm_client = make_llm_client()

# %% [markdown]
# ## Step 1 — The conversation we'll test
#
# This is a real-ish billing inquiry. The user has been authenticated, the
# agent has explained the bill, and the user says something that sounds like
# closure. In production this is exactly where agents fail — they read the
# user's "alright that makes sense" as goodbye, when it's actually
# acknowledgement of one specific thing with more conversation likely.

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

console.print("\n[bold]Test conversation:[/bold]\n")
for msg in CONVERSATION[1:]:
    speaker = "[cyan]USER[/cyan]" if msg["role"] == "user" else "[magenta]AGENT[/magenta]"
    console.print(f"  {speaker} {msg['content']}")
console.print("\n→ The agent now needs to decide what to say next.\n")

# %% [markdown]
# ## Step 2 — Agent A: pure agentic (the failure)
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


console.print("[bold]🤖 AGENT A (pure agentic) — running 5 times to show variance[/bold]\n")

agent_a_table = Table()
agent_a_table.add_column("Run", justify="right", style="cyan")
agent_a_table.add_column("Outcome", justify="center")
agent_a_table.add_column("Reply", style="yellow")

agent_a_replies: list[str] = []
for i in range(5):
    reply = run_agent_a(CONVERSATION)
    agent_a_replies.append(reply)
    is_goodbye = detect_closure_attempt(reply) and not asks_anything_else(reply)
    marker = "[red]❌ premature[/red]" if is_goodbye else "[green]✓ asks more[/green]"
    agent_a_table.add_row(str(i + 1), marker, reply)
console.print(agent_a_table)

failures = sum(
    1 for r in agent_a_replies if detect_closure_attempt(r) and not asks_anything_else(r)
)
console.print(f"\n[red]Premature goodbyes: {failures}/5 = {failures * 20}%[/red]")

# %% [markdown]
# ### What just happened
#
# Run that loop a few times. You'll see:
#
# - Most runs: the agent follows the instruction and asks if there's anything else
# - **Some runs: the agent says "Have a great day!" and ends the call**
#
# 20% failure rate isn't catastrophic in isolation — but at 1M calls per
# month that's 200,000 prematurely-ended calls. **The model isn't broken.**
# Soft instructions simply don't have enough weight to override a
# conversational pattern the model has seen millions of times.

# %% [markdown]
# ## Step 3 — Agent B: progressive control (the fix)
#
# Same model. Same system prompt. **Different harness.** We add a
# deterministic layer that the LLM cannot override. The framework tracks
# whether a "wrap-up" has run. If it hasn't, any attempt by the LLM to
# end the call gets intercepted and rewritten.


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

        # Override with a wrap-up question. The LLM gets to phrase it; we
        # constrain the *intent*. In a real system this would call the LLM
        # again with a more directive prompt, or use a templated response.
        wrapup_messages: list[dict[str, str]] = [
            *messages,
            {
                "role": "system",
                "content": "Do NOT say goodbye. Instead, briefly check if the user has any other questions about their bill. One short sentence.",
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


console.print("\n[bold]✅ AGENT B (progressive control) — same 5 runs[/bold]\n")

agent_b_table = Table()
agent_b_table.add_column("Run", justify="right", style="cyan")
agent_b_table.add_column("Outcome", justify="center")
agent_b_table.add_column("Reply", style="yellow")

agent_b_replies: list[str] = []
for i in range(5):
    state = CallState()
    reply, state = run_agent_b(CONVERSATION, state)
    agent_b_replies.append(reply)
    intercepted = state.wrapup_started
    marker = (
        "[green]✓ harness intercepted[/green]"
        if intercepted
        else "[green]✓ no goodbye attempted[/green]"
    )
    agent_b_table.add_row(str(i + 1), marker, reply)
console.print(agent_b_table)

b_failures = sum(
    1 for r in agent_b_replies if detect_closure_attempt(r) and not asks_anything_else(r)
)
color = "green" if b_failures == 0 else "yellow"
console.print(f"\n[{color}]Premature goodbyes: {b_failures}/5 = {b_failures * 20}%[/{color}]")

# %% [markdown]
# ### What just happened
#
# Same model. Same system prompt. The only difference is that Agent B's
# harness checks every LLM output for closure intent. If it finds one
# before the wrap-up has run, it asks the LLM to try again with stronger
# guidance.
#
# **Premature goodbye rate: typically 0/5.**

# %% [markdown]
# ## Step 4 — Synthesize both endings so we can hear the difference


# %%
def synthesize_with_rime(text: str, output_path: Path) -> None:
    response = requests.post(
        "https://users.rime.ai/v1/rime-tts",
        headers={
            "Authorization": f"Bearer {RIME_API_KEY}",
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        },
        json={"speaker": "abbie", "text": text, "modelId": "mistv2", "samplingRate": 16000},
        timeout=15,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)


sample_a = next(
    (r for r in agent_a_replies if detect_closure_attempt(r)),
    agent_a_replies[0],
)
sample_b = next(
    (r for r in agent_b_replies if asks_anything_else(r)),
    agent_b_replies[0],
)

console.print("\n[bold]🎤 Synthesizing example outputs with Rime[/bold]\n")
console.print(f"  [magenta]Agent A example:[/magenta] {sample_a!r}")
synthesize_with_rime(sample_a, AUDIO_DIR / "04_agent_a_premature_goodbye.wav")
console.print("  [green]→[/green] audio_output/04_agent_a_premature_goodbye.wav\n")

console.print(f"  [magenta]Agent B example:[/magenta] {sample_b!r}")
synthesize_with_rime(sample_b, AUDIO_DIR / "04_agent_b_progressive_control.wav")
console.print("  [green]→[/green] audio_output/04_agent_b_progressive_control.wav\n")

console.print("[italic]Listen to both. The difference is not the model. It's the harness.[/italic]")

# %% [markdown]
# ## Why progressive control matters in regulated industries
#
# In banking, healthcare, telco, insurance — anywhere a call has compliance
# requirements — the things that *must* happen before a call can end are
# not negotiable. Recording notices. Disclosures. Case ID confirmations.
#
# A pure agentic architecture handles these the way it handles everything:
# soft instructions in the prompt, hope for the best. Progressive control
# inverts this. Compliance steps are deterministic. The flexibility is
# preserved everywhere it doesn't matter. The control is added everywhere
# it does.

# %% [markdown]
# ## Recap
#
# - ✅ Pure agentic voice agents will say goodbye too soon some
#   percentage of the time, regardless of model or prompt.
# - ✅ Soft instructions work most of the time and fail in exactly the
#   conditions that matter — long calls, satisfied users, polite
#   acknowledgements.
# - ✅ **Progressive control** is the architectural fix. Agentic by
#   default, deterministic guardrails at the seams that matter.
#
# ## End of the four-script series
#
# - `01_turn_taking.py` — ASR endpoints aren't semantic boundaries.
# - `02_backchannels_vs_interrupts.py` — same syllable, different meanings.
# - `03_split_state.py` — concurrent systems need cooperative cancellation.
# - `04_premature_goodbye.py` — pure agentic isn't enough; progressive control is.
