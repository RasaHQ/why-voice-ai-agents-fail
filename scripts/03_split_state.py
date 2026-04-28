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
# # Failure 03 — The system disagrees with itself
#
# **Why (Voice) AI Agents Fail · Failure mode #3**
#
# > User says "stop". Three components see it. They disagree about what to do.
#
# Voice agents are concurrent systems. While the agent is "thinking" — running
# tool calls, generating responses, calling models — the user can speak. Audio
# can play. State changes. If your dialogue framework was designed for chat
# (where the user hits send and waits), this concurrency will surface as a
# specific failure: the agent finishes work the user already canceled, and
# replies to a question that's now stale.
#
# **What you'll see:**
#
# 1. We build a tiny voice agent with three concurrent components: ASR (mock,
#    fed by **Deepgram**-style transcripts), agent loop (real LLM call), and
#    TTS (real **Rime** synthesis).
# 2. We simulate a user barge-in mid-tool-call.
# 3. We run two versions: one without cancellation (the failure), one with
#    cooperative cancellation (the fix).
# 4. We measure: time-to-stale-reply, audio-leakage, user-experienced latency.
#
# **Total runtime:** ~30 seconds. **Cost:** ~$0.005.
#
# Run from the repo root:
#
# ```bash
# make script-03
# ```

# %%
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

import openai
import requests
from rich.panel import Panel
from rich.table import Table

from scripts._utils import console, ensure_audio_dir, get_key

# %% [markdown]
# ## Setup

# %%
RIME_API_KEY = get_key("RIME_API_KEY")
OPENAI_API_KEY = get_key("OPENAI_API_KEY")
AUDIO_DIR = ensure_audio_dir()

console.print(Panel.fit("[bold cyan]Failure 03 — The system disagrees with itself[/bold cyan]"))

# %% [markdown]
# ## Step 1 — The cancellation primitive

# %%
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)


@dataclass
class CancellationToken:
    """Cooperative cancellation primitive.

    The agent loop checks `cancelled` between every step. When the channel
    detects a barge-in, it flips the flag. In-flight work that hasn't yet
    reached a check point will complete (we can't kill an HTTP request
    mid-flight), but no new work starts.
    """

    cancelled: bool = False
    cancelled_at: float | None = None

    def cancel(self) -> None:
        if not self.cancelled:
            self.cancelled = True
            self.cancelled_at = time.time()

    def check(self) -> None:
        if self.cancelled:
            raise asyncio.CancelledError("Barge-in: user interrupted")


@dataclass
class AgentEvent:
    """Anything that happens in the agent loop, with a timestamp."""

    component: str
    event: str
    detail: str
    t: float = field(default_factory=time.time)


# %% [markdown]
# ## Step 2 — The agent loop with optional cancellation
#
# Real production agent loops do roughly this on every turn: receive a
# transcript, optionally call a tool, call an LLM, generate a spoken response,
# send it to TTS. Each is a place where, if the user has just barged in, we
# want to stop. The `token.check()` calls are the cooperative cancellation
# pattern.


# %%
async def mock_tool_call(
    name: str,
    duration: float,
    token: CancellationToken | None,
    events: list[AgentEvent],
) -> str:
    """Simulate a slow tool call. Sleeps in increments and checks the token."""
    events.append(AgentEvent("agent", "tool_call_start", name))
    steps = 10
    step_duration = duration / steps
    for _ in range(steps):
        if token:
            token.check()
        await asyncio.sleep(step_duration)
    events.append(AgentEvent("agent", "tool_call_end", name))
    return f"<result of {name}>"


async def llm_response(
    transcript: str,
    tool_result: str,
    token: CancellationToken | None,
    events: list[AgentEvent],
) -> str:
    """Call the LLM with the tool result and generate a spoken response.

    Real LLM calls cannot be cancelled mid-request, so this is a check-before,
    check-after pattern.
    """
    if token:
        token.check()
    events.append(AgentEvent("agent", "llm_call_start", transcript[:30]))

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Reply in 1 short sentence suitable for speech."},
            {
                "role": "user",
                "content": f"User said: {transcript}\nTool result: {tool_result}\nReply briefly.",
            },
        ],
        max_tokens=40,
    )
    reply = (response.choices[0].message.content or "").strip()
    events.append(AgentEvent("agent", "llm_call_end", reply[:40]))

    if token:
        token.check()
    return reply


def synthesize_with_rime_sync(text: str, output_path: Path) -> float:
    """Synthesize TTS audio with Rime. Returns time taken."""
    t0 = time.time()
    response = requests.post(
        "https://users.rime.ai/v1/rime-tts",
        headers={
            "Authorization": f"Bearer {RIME_API_KEY}",
            "Accept": "audio/wav",
            "Content-Type": "application/json",
        },
        json={
            "speaker": "abbie",
            "text": text,
            "modelId": "mistv2",
            "samplingRate": 16000,
        },
        timeout=15,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)
    return time.time() - t0


async def agent_turn(
    transcript: str,
    token: CancellationToken | None,
    events: list[AgentEvent],
) -> str | None:
    """One full agent turn: tool call, LLM response, TTS. Returns reply or None."""
    try:
        tool_result = await mock_tool_call("lookup_account", 1.5, token, events)
        reply_text = await llm_response(transcript, tool_result, token, events)

        if token:
            token.check()
        events.append(AgentEvent("agent", "tts_start", reply_text[:30]))
        audio_path = AUDIO_DIR / f"03_agent_reply_{int(time.time() * 1000)}.wav"

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, synthesize_with_rime_sync, reply_text, audio_path)

        if token:
            token.check()
        events.append(AgentEvent("agent", "tts_end", str(audio_path)))
        events.append(AgentEvent("agent", "reply_sent", reply_text))
        return reply_text

    except asyncio.CancelledError:
        events.append(AgentEvent("agent", "cancelled", "stopped on barge-in"))
        return None


# %% [markdown]
# ## Step 3 — The user's barge-in
#
# We simulate a user saying "stop" 800ms into the agent's turn. In a real
# system this would come from Deepgram's streaming transcription firing a
# `NewTranscript` event mid-bot-speech.


# %%
async def simulate_user_bargein(
    token: CancellationToken | None,
    events: list[AgentEvent],
    delay: float = 0.8,
) -> None:
    """User says 'stop' after `delay` seconds."""
    await asyncio.sleep(delay)
    events.append(AgentEvent("channel", "barge_in_detected", "user said: stop"))
    if token:
        events.append(AgentEvent("channel", "cancellation_signaled", ""))
        token.cancel()
    else:
        events.append(AgentEvent("channel", "cancellation_NOT_signaled", "no token wired up"))


def render_events(events: list[AgentEvent], title: str) -> None:
    """Pretty-print the event log as a Rich table."""
    if not events:
        return
    start_t = events[0].t
    table = Table(title=title, show_header=True)
    table.add_column("t", justify="right", style="dim")
    table.add_column("Component", style="cyan")
    table.add_column("Event")
    table.add_column("Detail", style="yellow")
    for e in events:
        rel_t = e.t - start_t
        table.add_row(f"{rel_t:5.2f}s", e.component, e.event, e.detail)
    console.print(table)


# %% [markdown]
# ## Step 4 — The failure: agent runs to completion despite barge-in


# %%
async def run_failure_demo() -> list[AgentEvent]:
    events = [AgentEvent("system", "scenario", "FAILURE: no cancellation")]
    transcript = "What's my account balance?"
    events.append(AgentEvent("user", "transcript_committed", transcript))

    agent_task = asyncio.create_task(agent_turn(transcript, None, events))
    bargein_task = asyncio.create_task(simulate_user_bargein(None, events))

    await asyncio.gather(agent_task, bargein_task)
    return events


console.print("\n[bold red]🚨 SCENARIO 1: No cancellation (the failure)[/bold red]\n")
failure_events = asyncio.run(run_failure_demo())
render_events(failure_events, "Without cancellation")

bargein_t = next((e.t for e in failure_events if e.event == "barge_in_detected"), None)
reply_t = next((e.t for e in failure_events if e.event == "reply_sent"), None)
if bargein_t and reply_t:
    stale_delay = reply_t - bargein_t
    console.print(f"\n[red]⚠ Time from barge-in to stale reply: {stale_delay:.2f}s[/red]")
    console.print(
        f"  The agent kept working for [bold]{stale_delay:.2f}s[/bold] after the user "
        f"said 'stop' and then played a reply to a question they'd moved on from."
    )

# %% [markdown]
# ### What just happened
#
# The user said "stop" at ~0.8s. The barge-in was detected by the channel —
# but there was no cancellation mechanism wired into the agent. The agent
# completed its 1.5-second tool call, called the LLM, called Rime for TTS,
# and produced an audio reply. **Total stale-reply delay: ~1.5-2.5 seconds
# of confusion.**

# %% [markdown]
# ## Step 5 — The fix: cooperative cancellation


# %%
async def run_fix_demo() -> list[AgentEvent]:
    events = [AgentEvent("system", "scenario", "FIX: cooperative cancellation")]
    transcript = "What's my account balance?"
    events.append(AgentEvent("user", "transcript_committed", transcript))

    token = CancellationToken()
    agent_task = asyncio.create_task(agent_turn(transcript, token, events))
    bargein_task = asyncio.create_task(simulate_user_bargein(token, events))

    await asyncio.gather(agent_task, bargein_task)
    return events


console.print("\n\n[bold green]✅ SCENARIO 2: Cooperative cancellation (the fix)[/bold green]\n")
fix_events = asyncio.run(run_fix_demo())
render_events(fix_events, "With cooperative cancellation")

bargein_t = next((e.t for e in fix_events if e.event == "barge_in_detected"), None)
cancelled_t = next((e.t for e in fix_events if e.event == "cancelled"), None)
if bargein_t and cancelled_t:
    response_delay = cancelled_t - bargein_t
    console.print(
        f"\n[green]✓ Time from barge-in to clean exit: {response_delay * 1000:.0f}ms[/green]"
    )
    console.print("  The agent stopped cleanly. No stale audio. No wasted LLM call.")

# %% [markdown]
# ## The architectural lesson
#
# What just changed wasn't the model. It wasn't the ASR. It wasn't the TTS.
# What changed is that **the agent loop knows how to be told to stop**.
#
# The pattern:
#
# 1. **Every long-running operation accepts a cancellation token.** Tool
#    calls, LLM calls, TTS synthesis. They check the token at well-defined
#    points.
# 2. **One canonical place owns the token.** When the channel detects a
#    barge-in, it flips the flag. Everything downstream sees it on the
#    next check.
# 3. **Cancellation is cooperative, not violent.** We don't kill threads
#    or yank HTTP requests. We let in-flight work complete to a clean state
#    and prevent new work from starting.
#
# Same pattern as `asyncio.CancelledError`, Go contexts, OS signals.
# **Voice agents are concurrent systems and need to be designed as such.**

# %% [markdown]
# ## Recap
#
# - ✅ Voice agents are **concurrent systems**. Things happen during the
#   agent's "turn" — interrupts, pauses, network jitter.
# - ✅ Without cooperative cancellation, the agent finishes work the user
#   already canceled and replies to a stale question.
# - ✅ The fix is a **cancellation token** that every long-running
#   operation checks between steps.
#
# **Next:** `04_premature_goodbye.py` — when the LLM has freedom over the
# words and decides the call is over while the user is still thinking.
