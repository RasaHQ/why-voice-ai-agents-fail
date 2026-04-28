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
# Run from the repo root: `make script-03`

# %%
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path

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
    "Failure 03 — The system disagrees with itself",
    "A voice agent is at least 3 concurrent components. "
    "When the user barges in, who knows about it?",
)

# %% [markdown]
# ## What we're about to do

# %%
watch_this(
    "I'll start an agent on a 1.5-second tool call.\n"
    "800ms in, the user will say 'STOP'. You'll hear it.\n"
    "Twice:\n"
    "  - First with NO cancellation wired up — agent ignores the user, plays a stale reply.\n"
    "  - Then WITH cooperative cancellation — agent stops cleanly in 100ms.",
)

# Pre-synthesize the user's "stop" once. We'll play it twice — once per scenario.
USER_STOP_AUDIO = AUDIO_DIR / "03_user_stop.mp3"
if not USER_STOP_AUDIO.exists():
    with live_status("Synthesizing the user's 'stop' utterance"):
        synthesize("Stop. Wait.", USER_STOP_AUDIO, audio_format="mp3", speed_alpha=1.1)


# %% [markdown]
# ## Cancellation primitive


# %%
@dataclass
class CancellationToken:
    """Cooperative cancellation primitive.

    Every long-running operation in the agent loop checks the token
    between steps. When the channel detects a barge-in, it flips the
    flag. New work doesn't start; in-flight work exits at the next check.
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
    """One thing that happened in the agent loop, with a timestamp."""

    component: str
    event: str
    detail: str
    t: float = field(default_factory=time.time)


# %% [markdown]
# ## The agent loop

# %%
llm_client = make_llm_client()


async def mock_tool_call(
    name: str,
    duration: float,
    token: CancellationToken | None,
    events: list[AgentEvent],
) -> str:
    """Simulate a slow tool call (e.g. account lookup). Sleeps in increments."""
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
    """Generate a spoken response. LLM calls aren't cancellable mid-request."""
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
    """TTS via Rime, blocking — wrapped by run_in_executor."""
    synthesize(text, output_path, audio_format="mp3")


async def agent_turn(
    transcript: str,
    token: CancellationToken | None,
    events: list[AgentEvent],
    audio_out: Path,
) -> str | None:
    """One full agent turn: tool call → LLM → TTS. Returns reply or None."""
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
    """User says 'stop' after `delay` seconds. We PLAY the audio so the audience hears it."""
    await asyncio.sleep(delay)
    events.append(AgentEvent("user", "barge_in", "user said: stop"))

    # Play the user's "stop" out loud, non-blocking — concurrently with the
    # agent's tool call so the audience hears the timing.
    play(USER_STOP_AUDIO, label="(user barges in)", blocking=False)

    if token:
        events.append(AgentEvent("channel", "cancel_signaled", ""))
        token.cancel()


# %% [markdown]
# ## Scenario 1 — No cancellation (the failure)

# %%
section("Scenario 1 — No cancellation wired up")
narrate(
    "Watch the timeline. User says 'stop' at 0.8s. The agent has no idea — "
    "it keeps running. Tool finishes. LLM runs. TTS runs. THEN the reply plays."
)
console.print()

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

# Compact event display — only the events that matter, with timing on the LEFT.
console.print()
start_t = failure_events[0].t
relevant = [e for e in failure_events if e.event in {"barge_in", "tool_call_end", "reply_sent"}]
for e in relevant:
    rel = e.t - start_t
    if e.event == "barge_in":
        console.print(f"  [bold red][{rel:5.2f}s][/bold red] [bold]👤 USER:[/bold] 'stop'")
    elif e.event == "tool_call_end":
        console.print(
            f"  [dim][{rel:5.2f}s] tool finished (agent didn't notice the interrupt)[/dim]"
        )
    elif e.event == "reply_sent":
        console.print(
            f'  [bold red][{rel:5.2f}s] 🤖 AGENT replies (STALE):[/bold red] "{e.detail}"'
        )

bargein_t = next((e.t for e in failure_events if e.event == "barge_in"), None)
reply_t = next((e.t for e in failure_events if e.event == "reply_sent"), None)
stale_delay = (reply_t - bargein_t) if (bargein_t and reply_t) else 0.0

# Now play the stale reply that the user hears AFTER they said stop.
if failure_audio.exists() and reply_t:
    pause_for_effect(0.3)
    play(failure_audio, label=f"Listen — stale reply, {stale_delay:.1f}s after the user said stop")

verdict(
    f"User said STOP at 0.8s. The agent kept working for {stale_delay:.1f} more seconds.",
    "By the time the reply plays, the user has moved on. They hear it as a broken bot.\n"
    "In production this is the moment they hang up.",
    kind="fail",
)

pause_for_effect(0.6)

# %% [markdown]
# ## Scenario 2 — Cooperative cancellation (the fix)

# %%
section("Scenario 2 — Same agent, but with a cancellation token")
narrate(
    "One change: every long-running step now checks a cancellation token. "
    "When the channel detects a barge-in, it flips the flag. Watch."
)
console.print()


async def run_fix_demo() -> list[AgentEvent]:
    events = [AgentEvent("system", "scenario", "FIX: cooperative cancellation")]
    transcript = "What's my account balance?"
    events.append(AgentEvent("user", "transcript_committed", transcript))
    token = CancellationToken()
    agent_task = asyncio.create_task(
        agent_turn(transcript, token, events, AUDIO_DIR / "03_clean_exit.mp3")
    )
    bargein_task = asyncio.create_task(simulate_user_bargein(token, events))
    await asyncio.gather(agent_task, bargein_task)
    return events


with live_status("Running scenario 2 — token wired through"):
    fix_events = asyncio.run(run_fix_demo())

console.print()
start_t = fix_events[0].t
fix_relevant = [e for e in fix_events if e.event in {"barge_in", "cancel_signaled", "cancelled"}]
for e in fix_relevant:
    rel = e.t - start_t
    if e.event == "barge_in":
        console.print(f"  [bold red][{rel:5.2f}s][/bold red] [bold]👤 USER:[/bold] 'stop'")
    elif e.event == "cancel_signaled":
        console.print(f"  [bold green][{rel:5.2f}s] ⚡ channel signals cancellation[/bold green]")
    elif e.event == "cancelled":
        console.print(f"  [bold green][{rel:5.2f}s] ✓ agent stopped cleanly[/bold green]")

bargein_t2 = next((e.t for e in fix_events if e.event == "barge_in"), None)
cancelled_t = next((e.t for e in fix_events if e.event == "cancelled"), None)
clean_delay = (cancelled_t - bargein_t2) if (bargein_t2 and cancelled_t) else 0.0

verdict(
    f"User said STOP at 0.8s. The agent stopped {int(clean_delay * 1000)}ms later.",
    "No stale audio. No wasted LLM call. No wasted TTS. The user gets the floor immediately.",
    kind="win",
)

# %% [markdown]
# ## Big comparison

# %%
big_compare(
    "Without cancellation",
    f"{stale_delay:.1f} seconds\nof stale audio\nplaying AFTER\n'stop'",
    "With cooperative\ncancellation",
    f"{int(clean_delay * 1000)} ms\nto a clean exit\nNo stale audio",
)

# %% [markdown]
# ## Takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]Voice agents are concurrent systems, not turn-based ones.[/bold]\n"
    "  [bold cyan]Every long-running operation needs to be cancellable.[/bold cyan]\n"
)

console.print()
console.print(
    "[bold magenta]Next:[/bold magenta]  [green]make script-04[/green]   [dim]— The bot says goodbye too soon[/dim]"
)
console.print()
