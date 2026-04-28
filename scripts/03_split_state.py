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
# Voice agents are concurrent systems. While the agent is "thinking", the
# user can speak. State changes. If your dialogue framework was designed
# for chat (where the user hits send and waits), this concurrency surfaces
# as a failure: the agent finishes work the user already cancelled.
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
    "Failure 03 — The system disagrees with itself",
    "A voice agent is at least 3 concurrent components. When the user barges in, "
    "who knows about it?",
)

narrate(
    "I'm going to barge in on a voice agent mid-tool-call. Twice. First with "
    "no cancellation wired up — you'll hear the stale reply. Second with "
    "cooperative cancellation — you'll hear the agent stop cleanly.",
    style="italic dim",
)

llm_client = make_llm_client()


# %% [markdown]
# ## The cancellation primitive


# %%
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
# ## The agent loop with optional cancellation


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

    Real LLM calls cannot be cancelled mid-request. Pattern: check before,
    check after.
    """
    if token:
        token.check()
    events.append(AgentEvent("agent", "llm_call_start", transcript[:30]))

    response = llm_client.chat.completions.create(
        model=DEFAULT_NEBIUS_MODEL,
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


def synthesize_reply_sync(text: str, output_path: Path) -> None:
    """TTS via Rime, blocking — wrapped by run_in_executor in the agent loop."""
    synthesize(text, output_path, audio_format="mp3")


async def agent_turn(
    transcript: str,
    token: CancellationToken | None,
    events: list[AgentEvent],
    audio_out: Path,
) -> str | None:
    """One full agent turn: tool call, LLM, TTS. Returns reply or None."""
    try:
        tool_result = await mock_tool_call("lookup_account", 1.5, token, events)
        reply_text = await llm_response(transcript, tool_result, token, events)

        if token:
            token.check()
        events.append(AgentEvent("agent", "tts_start", reply_text[:30]))

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, synthesize_reply_sync, reply_text, audio_out)

        if token:
            token.check()
        events.append(AgentEvent("agent", "tts_end", str(audio_out.name)))
        events.append(AgentEvent("agent", "reply_sent", reply_text))
        return reply_text

    except asyncio.CancelledError:
        events.append(AgentEvent("agent", "cancelled", "stopped on barge-in"))
        return None


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
    table.add_column("t", justify="right", style="dim", width=6)
    table.add_column("Component", style="cyan", width=10)
    table.add_column("Event", style="white")
    table.add_column("Detail", style="yellow")
    for e in events:
        rel_t = e.t - start_t
        table.add_row(f"{rel_t:.2f}s", e.component, e.event, e.detail)
    console.print(table)


# %% [markdown]
# ## Scenario 1 — No cancellation (the failure)

# %%
section("Scenario 1 — No cancellation (the failure)")

step(1, "User says 'What's my balance?' — agent starts working")
narrate(
    "The agent kicks off a 1.5s tool call. 800ms in, the user says 'stop'. "
    "There's no cancellation wired up. Watch what happens.",
)

failure_audio = AUDIO_DIR / "03_stale_reply.mp3"


async def run_failure_demo() -> list[AgentEvent]:
    events = [AgentEvent("system", "scenario", "FAILURE: no cancellation")]
    transcript = "What's my account balance?"
    events.append(AgentEvent("user", "transcript_committed", transcript))

    agent_task = asyncio.create_task(agent_turn(transcript, None, events, failure_audio))
    bargein_task = asyncio.create_task(simulate_user_bargein(None, events))
    await asyncio.gather(agent_task, bargein_task)
    return events


with live_status("Running scenario 1 — agent has no token to check"):
    failure_events = asyncio.run(run_failure_demo())

render_events(failure_events, "Without cancellation")

bargein_t = next((e.t for e in failure_events if e.event == "barge_in_detected"), None)
reply_t = next((e.t for e in failure_events if e.event == "reply_sent"), None)
if bargein_t and reply_t:
    stale_delay = reply_t - bargein_t
    console.print()
    if failure_audio.exists():
        play(
            failure_audio,
            label=f"Listen — the reply that played {stale_delay:.1f}s after 'stop'",
        )
    punchline(
        f"User said 'stop' at 0.8s. Agent replied at {(reply_t - failure_events[0].t):.2f}s.\n"
        f"That's {stale_delay:.1f} seconds of 'is this thing broken?'\n"
        f"In production this is the moment users hang up.",
        kind="fail",
    )

pause_for_effect(0.8)

# %% [markdown]
# ## Scenario 2 — Cooperative cancellation (the fix)

# %%
section("Scenario 2 — Cooperative cancellation (the fix)")

narrate(
    "Same agent, same tools, same LLM. The only thing that changes: every "
    "long-running operation now accepts a cancellation token and checks it "
    "between steps. When the user barges in, the channel flips the flag.",
)

fix_audio = AUDIO_DIR / "03_clean_exit.mp3"


async def run_fix_demo() -> list[AgentEvent]:
    events = [AgentEvent("system", "scenario", "FIX: cooperative cancellation")]
    transcript = "What's my account balance?"
    events.append(AgentEvent("user", "transcript_committed", transcript))

    token = CancellationToken()
    agent_task = asyncio.create_task(agent_turn(transcript, token, events, fix_audio))
    bargein_task = asyncio.create_task(simulate_user_bargein(token, events))
    await asyncio.gather(agent_task, bargein_task)
    return events


with live_status("Running scenario 2 — token wired through"):
    fix_events = asyncio.run(run_fix_demo())

render_events(fix_events, "With cooperative cancellation")

bargein_t = next((e.t for e in fix_events if e.event == "barge_in_detected"), None)
cancelled_t = next((e.t for e in fix_events if e.event == "cancelled"), None)
if bargein_t and cancelled_t:
    response_delay = cancelled_t - bargein_t
    punchline(
        f"User said 'stop' at 0.8s. Agent stopped at {(cancelled_t - fix_events[0].t):.2f}s.\n"
        f"That's {response_delay * 1000:.0f}ms to a clean exit.\n"
        f"No stale audio. No wasted LLM call. No 'is this thing broken'.",
        kind="win",
    )

# %% [markdown]
# ## The takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]Voice agents are concurrent systems, not turn-based ones.[/bold]\n"
    "  [bold cyan]Every long-running operation needs to be cancellable.[/bold cyan]\n"
)
console.print(
    "  [dim]Same pattern as `asyncio.CancelledError`, Go contexts, OS signals.[/dim]\n"
    "  [dim]If your dialogue framework was designed for chat, this is a retrofit.[/dim]\n"
    "  [dim]Three components, one canonical token, no stale replies.[/dim]\n"
)

console.print()
console.print(
    "[bold magenta]Next:[/bold magenta]  make script-04   [dim]— The bot says goodbye too soon[/dim]"
)
console.print()
