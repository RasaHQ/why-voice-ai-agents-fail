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
# # Failure 01 — Turn-taking is harder than you think
#
# **Why (Voice) AI Agents Fail · Failure mode #1**
#
# > The user says their phone number. The system commits to it after the third digit.
#
# Run from the repo root: `make script-01`

# %%
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

from deepgram import DeepgramClient, FileSource, PrerecordedOptions

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
    section,
    step,
    synthesize,
    verdict,
    watch_this,
)

# %% [markdown]
# ## Setup

# %%
DEEPGRAM_API_KEY = get_key("DEEPGRAM_API_KEY")
NEBIUS_API_KEY = get_key("NEBIUS_API_KEY", required=False)
get_key("RIME_API_KEY")
AUDIO_DIR = ensure_audio_dir()

header(
    "Failure 01 — Turn-taking is harder than you think",
    "ASR splits speech on silence. Humans split it on meaning. They are not the same thing.",
)

# %% [markdown]
# ## What we're about to do

# %%
watch_this(
    "I'll have someone read out a phone number — with a normal breath in the middle.\n"
    "In a streaming voice agent, Deepgram emits a finalized transcript every time\n"
    "it hears a pause. So our user breathing → Deepgram emits TWICE.\n"
    "An agent built on raw Deepgram output commits to the FIRST half.\n"
    "Then I'll show the fix: a tiny LLM call between Deepgram and the agent.",
)

# %% [markdown]
# ## Step 1 — Generate the user's audio in two halves

# %%
PART_1_TEXT = "My number is zero, six, six, four..."
PART_2_TEXT = "...five, four, six, seven, five, six, zero."
BREATH_SECONDS = 1.0  # the audible pause between halves

PART_1_AUDIO = AUDIO_DIR / "01_phone_part1.mp3"
PART_2_AUDIO = AUDIO_DIR / "01_phone_part2.mp3"

step(1, "Synthesize the two halves with Rime")
console.print(
    f'  [italic]"{PART_1_TEXT}  '
    f'[bold yellow]<{BREATH_SECONDS:.0f}s breath>[/bold yellow]  '
    f'{PART_2_TEXT}"[/italic]'
)

with live_status("Asking Rime to speak each half"):
    synthesize(PART_1_TEXT, PART_1_AUDIO, audio_format="mp3")
    synthesize(PART_2_TEXT, PART_2_AUDIO, audio_format="mp3")

console.print("  [green]✓[/green] Both halves written")

# %% [markdown]
# ## Step 2 — Listen. The breath is real. The agent is about to mishear it.

# %%
console.print()
console.print("  [bold magenta]🔊 Listen — what a real user actually sounds like[/bold magenta]")
play(PART_1_AUDIO, blocking=True, label=None)
time.sleep(BREATH_SECONDS)  # the audible breath
play(PART_2_AUDIO, blocking=True, label=None)
pause_for_effect()

# %% [markdown]
# ## Step 3 — Send each half to Deepgram (simulating streaming)

# %%
step(2, "Hand each half to Deepgram, the way streaming voice agents do")
narrate(
    "In production, Deepgram streaming emits a finalized transcript "
    "every time it detects a pause longer than ~400ms. Our user breathed → "
    "two transcripts. We simulate that here by transcribing each half."
)


@dataclass
class TranscriptSegment:
    """One finalized transcript from Deepgram."""

    text: str
    confidence: float


def transcribe_one(audio_path: Path) -> TranscriptSegment:
    """Transcribe a single MP3 file and return its top alternative."""
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    with audio_path.open("rb") as f:
        buffer_data = f.read()

    payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp3"}
    options = PrerecordedOptions(model="nova-2", smart_format=True)

    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    response_dict = json.loads(response.to_json())

    alt = response_dict["results"]["channels"][0]["alternatives"][0]
    return TranscriptSegment(
        text=alt.get("transcript", ""),
        confidence=alt.get("confidence", 0.0),
    )


with live_status("Transcribing the two halves"):
    seg1 = transcribe_one(PART_1_AUDIO)
    seg2 = transcribe_one(PART_2_AUDIO)

segments = [seg1, seg2]

console.print()
console.print(
    f"  Deepgram emitted [bold red]{len(segments)} finalized transcripts:[/bold red]"
)
for i, seg in enumerate(segments, 1):
    console.print(f"    [cyan]{i}.[/cyan] [yellow]\"{seg.text}\"[/yellow]")
pause_for_effect(0.8)

# %% [markdown]
# ## What just happened

# %%
verdict(
    "The agent acts on the FIRST transcript and never sees the second.",
    f'It commits to looking up an account ending in "{seg1.text.split()[-1] if seg1.text else "???"}" — '
    f"finds nothing — and asks the user to start over.\n"
    f"The user's full phone number was right there in transcript #2. "
    f"The agent never got the chance.",
    kind="fail",
)

# %% [markdown]
# ## Step 3 — The fix: a tiny LLM call between ASR and agent

# %%
section("Now the fix")

if not NEBIUS_API_KEY:
    narrate(
        "[yellow]⚠ NEBIUS_API_KEY not set — skipping the fix demo.[/yellow]"
    )
else:
    narrate(
        "We layer a small LLM call between Deepgram and the agent. "
        "It answers one question: 'Is this a complete thought, or is the user still talking?'"
    )

    llm_client = make_llm_client()

    SEMANTIC_TURN_PROMPT = """You analyze partial transcripts from a voice agent.
Your job: decide whether the user has finished their thought or is mid-sentence.

Output exactly one word:
- COMPLETE — the user has finished and the agent should respond
- INCOMPLETE — the user is still talking, the agent should wait

Examples:
"My number is zero six six four" → INCOMPLETE
"Yes, that's correct." → COMPLETE
"I'd like to" → INCOMPLETE
"Can I check my balance?" → COMPLETE
"Hold on, let me find" → INCOMPLETE

Transcript: {transcript}
Output:"""

    def is_complete_thought(transcript: str) -> tuple[bool, float]:
        """Classify whether a transcript is a complete thought."""
        t0 = time.time()
        response = llm_client.chat.completions.create(
            model=DEFAULT_NEBIUS_MODEL,
            messages=[{"role": "user", "content": SEMANTIC_TURN_PROMPT.format(transcript=transcript)}],
            temperature=0,
            max_tokens=4,
        )
        latency = time.time() - t0
        word = (response.choices[0].message.content or "").strip().upper()
        return word.startswith("COMPLETE"), latency

    step(3, "Run each Deepgram segment through the LLM as it arrives")
    console.print()

    accumulated = ""
    final_transcript = ""
    for i, seg in enumerate(segments, 1):
        candidate = (accumulated + " " + seg.text).strip()
        with live_status(f"Asking Llama: is '{candidate[:40]}…' a complete thought?"):
            complete, latency = is_complete_thought(candidate)

        verdict_label = (
            "[green]COMPLETE → respond now[/green]" if complete
            else "[yellow]INCOMPLETE → wait for more[/yellow]"
        )
        console.print(
            f"  [cyan]{i}.[/cyan] [yellow]\"{candidate[:60]}\"[/yellow]  "
            f"→ {verdict_label}  [dim]({latency * 1000:.0f}ms)[/dim]"
        )
        if complete:
            final_transcript = candidate
            break
        accumulated = candidate

    if final_transcript:
        verdict(
            "The agent now sees the FULL phone number.",
            f'It commits on:  "{final_transcript}"\n'
            f"No restart. No frustrated user. Trust intact.",
            kind="win",
        )
    else:
        verdict(
            "All segments came back INCOMPLETE.",
            "In production you'd commit after a max-wait timer (e.g. 2s). "
            "The point holds: the LLM correctly resisted committing too early.",
            kind="info",
        )

# %% [markdown]
# ## Takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]ASR ends a turn on silence.[/bold]\n"
    "  [bold]A human ends a turn on meaning.[/bold]\n"
    "  [bold cyan]A tiny LLM layer turns the first into the second.[/bold cyan]\n"
)

console.print()
console.print("[bold magenta]Next:[/bold magenta]  [green]make script-02[/green]   [dim]— Backchannels vs interrupts[/dim]")
console.print()
