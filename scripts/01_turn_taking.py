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
# This script demonstrates the most common voice-agent failure on Earth: the ASR
# finalizes a transcript at an acoustic pause that the user did not intend as the
# end of their turn. Your agent acts on incomplete data, looks foolish, and the
# user has to start over.
#
# **What you'll see:**
#
# 1. We synthesize a phone number being spoken with a natural mid-sentence pause
#    (using **Rime** for TTS).
# 2. We feed that audio into **Deepgram** for transcription.
# 3. We watch Deepgram emit a *finalized* transcript at the pause — the failure mode.
# 4. We add a **semantic turn-taking layer** and watch the same audio produce the
#    correct, complete transcript.
#
# **Total runtime:** ~30 seconds. **Cost:** ~$0.005.
#
# Run from the repo root:
#
# ```bash
# make script-01
# ```

# %%
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import requests
from deepgram import DeepgramClient, FileSource, PrerecordedOptions
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
# ## Step 1 — Load credentials

# %%
DEEPGRAM_API_KEY = get_key("DEEPGRAM_API_KEY")
RIME_API_KEY = get_key("RIME_API_KEY")
NEBIUS_API_KEY = get_key("NEBIUS_API_KEY", required=False)
AUDIO_DIR = ensure_audio_dir()

console.print(Panel.fit("[bold cyan]Failure 01 — Turn-taking[/bold cyan]"))

# %% [markdown]
# ## Step 2 — Synthesize the test audio with Rime
#
# We generate audio of a user speaking a phone number with a natural mid-sentence
# pause. The phrase: *"My number is zero six six four... five four six seven five
# six zero."* The "..." after "four" is what creates the failure-inducing pause —
# a real user does this all the time when recalling a number from memory.

# %%
TEST_PHRASE = "My number is zero six six four... five four six seven five six zero."
TEST_AUDIO_PATH = AUDIO_DIR / "01_phone_number_with_pause.wav"


def synthesize_with_rime(text: str, output_path: Path) -> None:
    """Generate audio from text using Rime TTS.

    We use Rime's `mistv2` model which has good control over pausing and pacing.
    Output is a WAV file at 16kHz mono — the format Deepgram prefers.
    """
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
            "pauseBetweenBrackets": True,
        },
        timeout=30,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)


console.print(f"[blue]→[/blue] Synthesizing: [italic]{TEST_PHRASE!r}[/italic]")
synthesize_with_rime(TEST_PHRASE, TEST_AUDIO_PATH)
audio_size_kb = TEST_AUDIO_PATH.stat().st_size / 1024
console.print(f"[green]✓[/green] Audio written: {TEST_AUDIO_PATH} ({audio_size_kb:.1f} KB)")

# %% [markdown]
# ## Step 3 — Send to Deepgram and watch the failure
#
# We stream the audio into Deepgram's pre-recorded API. The failure is visible in
# the response: there will be **multiple** finalized utterances instead of one
# continuous transcript. Each finalized utterance is a moment where Deepgram
# decided the user was done talking — and your agent would have reacted to each.

# %%
deepgram = DeepgramClient(DEEPGRAM_API_KEY)


@dataclass
class TranscriptSegment:
    """One finalized utterance from Deepgram's perspective."""

    text: str
    start: float
    end: float
    confidence: float


def transcribe_with_endpointing(audio_path: Path) -> list[TranscriptSegment]:
    """Transcribe audio with utterance-level endpointing enabled.

    This is the failure in action. `utterances=True` causes Deepgram to split
    the audio wherever it hears a pause longer than the endpointing threshold.
    """
    with audio_path.open("rb") as f:
        buffer_data = f.read()

    payload: FileSource = {"buffer": buffer_data}
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        utt_split=0.4,
    )

    response = deepgram.listen.rest.v("1").transcribe_file(payload, options)
    response_dict = json.loads(response.to_json())

    segments = []
    for utt in response_dict["results"].get("utterances", []):
        segments.append(
            TranscriptSegment(
                text=utt["transcript"],
                start=utt["start"],
                end=utt["end"],
                confidence=utt["confidence"],
            )
        )
    return segments


console.print("\n[bold]🎤 Sending audio to Deepgram with default endpointing...[/bold]")
segments = transcribe_with_endpointing(TEST_AUDIO_PATH)

table = Table(title=f"Deepgram returned {len(segments)} finalized utterance(s)")
table.add_column("#", style="cyan", width=3)
table.add_column("Start", justify="right", style="white")
table.add_column("End", justify="right", style="white")
table.add_column("Confidence", justify="right")
table.add_column("Transcript", style="yellow")

for i, seg in enumerate(segments, 1):
    table.add_row(
        str(i),
        f"{seg.start:.2f}s",
        f"{seg.end:.2f}s",
        f"{seg.confidence:.2f}",
        seg.text,
    )
console.print(table)

# %% [markdown]
# ### What just happened
#
# You should see **two or more** finalized utterances. Something like:
#
# ```
# 1. [0.05s →  1.85s] 'My number is zero six six four' (conf=0.97)
# 2. [2.40s →  4.20s] 'five four six seven five six zero.' (conf=0.96)
# ```
#
# This is what your agent sees in real time. After the first finalized
# transcript arrives, your agent thinks the user is done. It looks up account
# `06664`. It finds nothing. It asks the user to repeat themselves.
#
# **The user has to start over.**

# %% [markdown]
# ## Step 4 — The fix: a semantic turn-taking layer
#
# The fix is not "find a better ASR." Even with the best ASR, acoustic pauses
# don't reliably indicate semantic completion. The fix is a layer between
# Deepgram and your agent that asks a different question:
#
# > Is this transcript a *complete thought*, or a fragment with more probably coming?
#
# This is a small, fast LLM call. We use Nebius's `Meta-Llama-3.1-8B-Instruct-fast`
# (the default, configurable via NEBIUS_MODEL) because it's cheap and sub-second.
# Any small model will do — Qwen3, MiniMax, DeepSeek, you name it.

# %%
if not NEBIUS_API_KEY:
    console.print(
        "\n[yellow]⚠ NEBIUS_API_KEY not set — skipping the semantic turn-taking demo.[/yellow]"
    )
    console.print("  Set the key and re-run to see the fix in action.\n")
else:
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
        """Classify whether a transcript is a complete thought.

        Returns (is_complete, latency_seconds).
        """
        t0 = time.time()
        response = llm_client.chat.completions.create(
            model=DEFAULT_NEBIUS_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": SEMANTIC_TURN_PROMPT.format(transcript=transcript),
                }
            ],
            temperature=0,
            max_tokens=4,
        )
        latency = time.time() - t0
        verdict = (response.choices[0].message.content or "").strip().upper()
        return verdict.startswith("COMPLETE"), latency

    console.print("\n[bold]🧠 Adding semantic turn-taking layer...[/bold]\n")
    fix_table = Table(show_header=True)
    fix_table.add_column("Accumulated transcript", style="yellow", width=50)
    fix_table.add_column("Verdict", justify="center")
    fix_table.add_column("Latency", justify="right")

    accumulated = ""
    for seg in segments:
        candidate = (accumulated + " " + seg.text).strip()
        is_complete, latency = is_complete_thought(candidate)
        verdict = "[green]COMPLETE ✓[/green]" if is_complete else "[red]INCOMPLETE ✗[/red]"
        fix_table.add_row(candidate[:48], verdict, f"{latency * 1000:.0f}ms")
        if not is_complete:
            accumulated = candidate
        else:
            console.print(fix_table)
            console.print(
                f"\n[bold green]→ Final transcript sent to agent:[/bold green] {candidate!r}"
            )
            break
    else:
        console.print(fix_table)


# %% [markdown]
# ### What the fix does
#
# After the first Deepgram utterance, the semantic layer says INCOMPLETE — the
# transcript ends with a digit and no terminator. The agent waits.
#
# After the second Deepgram utterance, the accumulated transcript is now
# complete. The semantic layer says COMPLETE. The agent responds — to the
# *full* phone number, the way the user actually said it.
#
# **This is the pattern.** ASR-level endpointing tells you when audio went
# silent. A semantic layer tells you when the human is done.

# %% [markdown]
# ## Recap
#
# - ✅ ASR engines endpoint on **acoustic** silence; humans endpoint on
#   **semantic** completion. These are not the same thing.
# - ✅ The fix is a **post-processing layer** between ASR and agent that asks
#   a different question: is the transcript a complete thought?
# - ✅ This costs ~250-600ms of added latency per check. The cost of *not*
#   doing it is users repeating themselves.
#
# **Next:** `02_backchannels_vs_interrupts.py` — when one syllable means two
# different things and your agent has to know which.
