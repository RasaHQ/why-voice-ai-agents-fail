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
# This is the most common voice failure on Earth. We're going to make it
# happen, in real time, with real Rime audio and a real Deepgram call.
# Then we'll fix it with a small LLM layer and run the same audio through.
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

from deepgram import DeepgramClient, FileSource, PrerecordedOptions
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
DEEPGRAM_API_KEY = get_key("DEEPGRAM_API_KEY")
NEBIUS_API_KEY = get_key("NEBIUS_API_KEY", required=False)
get_key("RIME_API_KEY")  # eager-checked, raises if missing
AUDIO_DIR = ensure_audio_dir()

header(
    "Failure 01 — Turn-taking is harder than you think",
    "ASR endpoints on acoustic silence. Humans endpoint on semantic completion. "
    "Same audio, different meaning.",
)

narrate(
    "I'm going to make a voice agent commit to a half-spoken phone number. "
    "Watch — and listen. The fix is one small LLM call.",
    style="italic dim",
)

# %% [markdown]
# ## Step 1 — Synthesize a phone number with a natural breath
#
# We use Rime to generate a phone number being spoken by a real human-sounding
# voice, with a natural mid-sentence pause after "four" — the kind of pause
# every user does when recalling a number from memory.

# %%
TEST_PHRASE = "My number is zero six six four... five four six seven five six zero."
TEST_AUDIO_PATH = AUDIO_DIR / "01_phone_number.mp3"

step(1, "Synthesize a phone number — with a natural breath in the middle")
console.print(f'  [italic dim]"{TEST_PHRASE}"[/italic dim]')

with live_status("Asking Rime to speak it"):
    synthesize(TEST_PHRASE, TEST_AUDIO_PATH, audio_format="mp3")

audio_size_kb = TEST_AUDIO_PATH.stat().st_size / 1024
console.print(f"  [green]✓[/green] {audio_size_kb:.1f} KB written")
play(TEST_AUDIO_PATH, label="Listen — what a real user sounds like")
pause_for_effect()

# %% [markdown]
# ## Step 2 — Hand the audio to Deepgram, the way every voice agent does
#
# This is the moment where production voice agents go wrong. Deepgram does
# its job perfectly: it turns audio into text. But its definition of "the user
# is done" is "audio went silent for a beat" — which doesn't match the human
# definition.

# %%
step(2, "Send to Deepgram — same way a production voice agent would")

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

    Note we pass `mimetype="audio/mp3"` explicitly. Deepgram can usually
    auto-detect format, but it occasionally fails on Rime's output —
    explicit is better.
    """
    with audio_path.open("rb") as f:
        buffer_data = f.read()

    payload: FileSource = {"buffer": buffer_data, "mimetype": "audio/mp3"}
    options = PrerecordedOptions(
        model="nova-2",
        smart_format=True,
        utterances=True,
        utt_split=0.4,  # split on pauses ≥400ms — Deepgram's default
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


with live_status("Transcribing with Deepgram nova-2"):
    segments = transcribe_with_endpointing(TEST_AUDIO_PATH)

table = Table(
    title=f"What Deepgram heard — [bold red]{len(segments)} separate utterance(s)[/bold red]",
    title_style="white",
)
table.add_column("#", style="cyan", width=3)
table.add_column("Start", justify="right")
table.add_column("End", justify="right")
table.add_column("Conf", justify="right")
table.add_column("Transcript", style="yellow")

for i, seg in enumerate(segments, 1):
    table.add_row(
        str(i),
        f"{seg.start:.2f}s",
        f"{seg.end:.2f}s",
        f"{seg.confidence:.2f}",
        f'"{seg.text}"',
    )
console.print()
console.print(table)
pause_for_effect()

# %% [markdown]
# ### What just happened

# %%
if len(segments) > 1:
    punchline(
        f"The user said ONE thing — their phone number.\n"
        f"The agent received {len(segments)} transcripts and would act on the first one.\n"
        f"It would look up account '{segments[0].text.split()[-1] if segments[0].text else '???'}', "
        f"find nothing, and ask the user to start over.",
        kind="fail",
    )
else:
    narrate(
        "Heads up: Deepgram returned only one segment this run. The split happens "
        "more reliably with longer pauses — try TEST_PHRASE with extra '...' to force it.",
        style="yellow",
    )

# %% [markdown]
# ## Step 3 — The fix: a semantic turn-taking layer
#
# The fix isn't a smarter ASR. It's a small, fast LLM call that asks a
# different question: "Is this transcript a complete thought?" Layer that
# between Deepgram and your agent and the failure goes away.

# %%
section("The fix — semantic turn-taking layer")

if not NEBIUS_API_KEY:
    narrate(
        "[yellow]⚠ NEBIUS_API_KEY not set — skipping the fix demo. "
        "Add it to .env to see the full pattern.[/yellow]"
    )
else:
    narrate(
        "We layer a small LLM call between Deepgram and the agent. The LLM "
        "answers one question: 'Is this a complete thought, or is the user "
        "still talking?' Cheap (~$0.0001), fast (~250-600ms), correct.",
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

    fix_table = Table(title="Semantic turn-taking layer", show_header=True)
    fix_table.add_column("Accumulated transcript", style="yellow", max_width=60)
    fix_table.add_column("Verdict", justify="center")
    fix_table.add_column("Latency", justify="right")

    accumulated = ""
    final_transcript = ""
    for seg in segments:
        candidate = (accumulated + " " + seg.text).strip()
        with live_status(f"Asking Llama: is '{candidate[:40]}...' complete?"):
            complete, latency = is_complete_thought(candidate)
        verdict = "[green]COMPLETE ✓[/green]" if complete else "[red]INCOMPLETE ⏸[/red]"
        fix_table.add_row(candidate[:60], verdict, f"{latency * 1000:.0f}ms")
        if complete:
            final_transcript = candidate
            break
        accumulated = candidate

    console.print(fix_table)

    if final_transcript:
        punchline(
            f"The agent now acts on the FULL transcript:\n"
            f'   "{final_transcript}"\n'
            f"No restart. No frustrated user. Trust intact.",
            kind="win",
        )
    else:
        narrate(
            "All segments were marked INCOMPLETE. In production you'd commit "
            "after a max wait timer expires.",
            style="yellow",
        )

# %% [markdown]
# ## The takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]ASR endpoints on acoustic silence.[/bold] That's its job.\n"
    "  [bold]Humans endpoint on semantic completion.[/bold] That's the real signal.\n"
    "  [bold cyan]A small LLM layer turns one into the other.[/bold cyan]\n"
)
console.print(
    "  [dim]200ms pause after 'zero six six' = breath.[/dim]\n"
    "  [dim]200ms pause after '...does that work?' = waiting for an answer.[/dim]\n"
    "  [dim]Same audio. Different meaning. Only the harness can tell.[/dim]\n"
)

console.print()
console.print(
    "[bold magenta]Next:[/bold magenta]  make script-02   [dim]— Backchannels vs interrupts[/dim]"
)
console.print()
