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
    synthesize_with_pause,
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
    "I'll record someone saying their phone number — with a normal breath in the middle.\n"
    "Deepgram (the ASR) will hear it as TWO things instead of one.\n"
    "An agent built on raw Deepgram output will commit to the FIRST half.\n"
    "Then I'll show the fix: a tiny LLM call between Deepgram and the agent.",
)

# %% [markdown]
# ## Step 1 — Generate the audio

# %%
TEST_PART_1 = "My number is zero, six, six, four..."
TEST_PART_2 = "five, four, six, seven, five, six, zero."
SILENCE_MS = 1000  # 1 full second — enough to force any reasonable ASR to endpoint
TEST_AUDIO_PATH = AUDIO_DIR / "01_phone_with_pause.wav"

step(1, "Generate the user audio (with a real 1-second breath in the middle)")
console.print(
    f'  [italic]"{TEST_PART_1}  [bold yellow]<1s breath>[/bold yellow]  {TEST_PART_2}"[/italic]'
)

with live_status("Asking Rime to speak both halves, splicing silence between them"):
    synthesize_with_pause(TEST_PART_1, TEST_PART_2, TEST_AUDIO_PATH, silence_ms=SILENCE_MS)

console.print(
    f"  [green]✓[/green] {TEST_AUDIO_PATH.stat().st_size / 1024:.0f} KB written "
    f"[dim]({TEST_AUDIO_PATH.name})[/dim]"
)
play(TEST_AUDIO_PATH, label="Listen — what a real user actually sounds like")
pause_for_effect()

# %% [markdown]
# ## Step 2 — Send it to Deepgram, the way every voice agent does

# %%
step(2, "Send to Deepgram with default settings (`utt_split=0.4` — 400ms)")
narrate(
    "This is the moment where production voice agents go wrong. "
    "Deepgram does its job perfectly — it just defines 'done' as 'audio went silent for a beat'."
)


@dataclass
class TranscriptSegment:
    """One finalized utterance from Deepgram's perspective."""

    text: str
    start: float
    end: float
    confidence: float


def transcribe_with_endpointing(audio_path: Path) -> list[TranscriptSegment]:
    """Transcribe with utterance-level endpointing enabled."""
    deepgram = DeepgramClient(DEEPGRAM_API_KEY)
    with audio_path.open("rb") as f:
        buffer_data = f.read()

    payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/wav"}
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        utt_split=0.4,
    )

    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    response_dict = json.loads(response.to_json())

    return [
        TranscriptSegment(
            text=u["transcript"],
            start=u["start"],
            end=u["end"],
            confidence=u["confidence"],
        )
        for u in response_dict["results"].get("utterances", [])
    ]


with live_status("Transcribing with Deepgram nova-2"):
    segments = transcribe_with_endpointing(TEST_AUDIO_PATH)

# Show ONLY the count and the first segment's text — that's the headline.
# We deliberately don't dump a full table here. Audience reads ONE thing.
console.print()
if len(segments) >= 2:
    console.print(
        f"  Deepgram heard the audio as [bold red]{len(segments)} separate transcripts:[/bold red]"
    )
    for i, seg in enumerate(segments, 1):
        console.print(f'    [cyan]{i}.[/cyan] [yellow]"{seg.text}"[/yellow]')
elif len(segments) == 1:
    console.print("  Deepgram heard the audio as [bold]1 transcript:[/bold]")
    console.print(f'    [yellow]"{segments[0].text}"[/yellow]')

pause_for_effect(0.8)

# %% [markdown]
# ## What just happened

# %%
if len(segments) >= 2:
    first_committed = segments[0].text
    verdict(
        "The agent acts on the FIRST transcript and never sees the rest.",
        f'It would look up an account ending in "{first_committed.split()[-1] if first_committed else "???"}" — '
        f"find nothing — and ask the user to repeat themselves.\n"
        f"The user starts over. Trust drops. The agent's intelligence never had a chance.",
        kind="fail",
    )
else:
    verdict(
        "Deepgram returned only one segment on this run.",
        "The pause wasn't long enough to trigger a split. "
        "Increase SILENCE_MS in the script and re-run, or lower utt_split.",
        kind="info",
    )

# %% [markdown]
# ## Step 3 — The fix: a tiny LLM call between ASR and agent

# %%
section("Now the fix")

if not NEBIUS_API_KEY:
    narrate(
        "[yellow]⚠ NEBIUS_API_KEY not set — skipping the fix demo. "
        "Set it in .env to see the rest.[/yellow]"
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
            messages=[
                {"role": "user", "content": SEMANTIC_TURN_PROMPT.format(transcript=transcript)}
            ],
            temperature=0,
            max_tokens=4,
        )
        latency = time.time() - t0
        verdict_word = (response.choices[0].message.content or "").strip().upper()
        return verdict_word.startswith("COMPLETE"), latency

    step(3, "Run each Deepgram segment through the LLM")
    console.print()

    accumulated = ""
    final_transcript = ""
    for i, seg in enumerate(segments, 1):
        candidate = (accumulated + " " + seg.text).strip()
        with live_status(f"Asking Llama: is '{candidate[:40]}...' a complete thought?"):
            complete, latency = is_complete_thought(candidate)

        verdict_label = (
            "[green]COMPLETE → respond now[/green]"
            if complete
            else "[yellow]INCOMPLETE → wait for more[/yellow]"
        )
        console.print(
            f'  [cyan]{i}.[/cyan] [yellow]"{candidate[:60]}"[/yellow]  '
            f"→ {verdict_label}  [dim]({latency * 1000:.0f}ms)[/dim]"
        )
        if complete:
            final_transcript = candidate
            break
        accumulated = candidate

    if final_transcript:
        verdict(
            "The agent now sees the FULL phone number.",
            f'It commits on:  "{final_transcript}"\nNo restart. No frustrated user. Trust intact.',
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
console.print(
    "[bold magenta]Next:[/bold magenta]  [green]make script-02[/green]   [dim]— Backchannels vs interrupts[/dim]"
)
console.print()
