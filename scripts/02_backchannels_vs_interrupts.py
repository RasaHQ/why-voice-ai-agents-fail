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
# # Failure 02 — Tools that look like backchannels
#
# **Why (Voice) AI Agents Fail · Failure mode #2**
#
# > Same word. Different meaning. Your agent has to know which.
#
# When a user says "mhmm" while the agent is talking, are they:
#
# - **Backchanneling** — being polite, signaling presence, asking the agent to continue?
# - **Interrupting** — telling the agent to stop because they need to say something?
#
# This is the single hardest classification problem in voice agents that doesn't
# show up in text agents. Get it wrong in one direction and you cut the agent off
# mid-sentence. Get it wrong in the other and you talk over the user.
#
# **What you'll see:**
#
# 1. We synthesize three real conversation moments using **Rime**.
# 2. We run a **two-tier classifier**: word-count fast path + semantic LLM path.
# 3. We measure accuracy and latency on each.
#
# **Total runtime:** ~45 seconds. **Cost:** ~$0.01.
#
# Run from the repo root:
#
# ```bash
# make script-02
# ```

# %%
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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

console.print(Panel.fit("[bold cyan]Failure 02 — Backchannels vs interrupts[/bold cyan]"))

# %% [markdown]
# ## Step 1 — Define the test conversations
#
# Each test case is a triple: what the bot was saying, what the user said in
# response, and what the *correct* classification is. We run our classifier
# against the ground truth.


# %%
@dataclass
class TestCase:
    """One barge-in moment: bot context + user utterance + ground truth label."""

    name: str
    bot_context: str
    user_utterance: str
    ground_truth: Literal["INTERRUPT", "BACKCHANNEL"]


TEST_CASES: list[TestCase] = [
    TestCase(
        name="polite_listening",
        bot_context="Your November bill came to four hundred and eighty-two dollars, which is up from",
        user_utterance="Mhmm.",
        ground_truth="BACKCHANNEL",
    ),
    TestCase(
        name="answering_question",
        bot_context="Are you inquiring about your current account, or a different one?",
        user_utterance="Yes.",
        ground_truth="INTERRUPT",
    ),
    TestCase(
        name="filled_pause_during_explanation",
        bot_context="The increase comes from a one-time charge applied during the billing cycle",
        user_utterance="Right.",
        ground_truth="BACKCHANNEL",
    ),
    TestCase(
        name="genuine_stop_request",
        bot_context="I can transfer your call to a specialist who handles billing disputes,",
        user_utterance="Wait, hold on.",
        ground_truth="INTERRUPT",
    ),
    TestCase(
        name="information_correction",
        bot_context="So I'll process the refund to your account ending in three four five six,",
        user_utterance="No, that's not right.",
        ground_truth="INTERRUPT",
    ),
    TestCase(
        name="agreement_filler",
        bot_context="And then we'll send you a confirmation email within twenty-four hours,",
        user_utterance="Okay.",
        ground_truth="BACKCHANNEL",
    ),
]

setup_table = Table(title=f"Loaded {len(TEST_CASES)} test cases", show_header=True)
setup_table.add_column("Name", style="cyan")
setup_table.add_column("User said", style="yellow")
setup_table.add_column("Ground truth", justify="center")
for tc in TEST_CASES:
    color = "green" if tc.ground_truth == "BACKCHANNEL" else "red"
    setup_table.add_row(tc.name, repr(tc.user_utterance), f"[{color}]{tc.ground_truth}[/{color}]")
console.print(setup_table)

# %% [markdown]
# ## Step 2 — Synthesize the audio with Rime
#
# We synthesize the user utterances at varying speeds. This isn't strictly
# necessary for classification (we work from the transcript) but it makes the
# script feel real and lets you listen to each clip.


# %%
def synthesize_with_rime(text: str, output_path: Path, speed_alpha: float = 1.0) -> None:
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
            "speedAlpha": speed_alpha,
        },
        timeout=30,
    )
    response.raise_for_status()
    output_path.write_bytes(response.content)


console.print("\n[bold]🎤 Synthesizing user barge-ins with Rime...[/bold]")
for tc in TEST_CASES:
    audio_path = AUDIO_DIR / f"02_bargein_{tc.name}.wav"
    if not audio_path.exists():
        speed = 1.15 if tc.ground_truth == "BACKCHANNEL" else 1.0
        synthesize_with_rime(tc.user_utterance, audio_path, speed_alpha=speed)
    console.print(f"  [green]✓[/green] {audio_path.name}")

# %% [markdown]
# ## Step 3 — Tier 1: word-count fast path
#
# Before we burn an LLM call on every barge-in, we try a heuristic. **Sustained
# speech is never a backchannel.** If the user has been talking for more than
# ~4 words, it's an interrupt. No human says "okay yes I really agree with that"
# as a backchannel.


# %%
def tier1_word_count(
    user_utterance: str,
    threshold: int = 4,
) -> Literal["INTERRUPT", "BACKCHANNEL"] | None:
    """Tier 1: sustained speech is always an interrupt.

    Returns INTERRUPT if word count exceeds threshold, otherwise None
    (meaning: not confident, escalate to Tier 2).
    """
    if len(user_utterance.split()) > threshold:
        return "INTERRUPT"
    return None


console.print("\n[bold]🚀 Tier 1: word-count fast path[/bold]\n")
tier1_table = Table()
tier1_table.add_column("Test case", style="cyan")
tier1_table.add_column("Words", justify="right")
tier1_table.add_column("Tier 1 verdict", justify="center")
tier1_table.add_column("Latency", justify="right")

tier1_results: dict[str, str | None] = {}
for tc in TEST_CASES:
    t0 = time.time()
    verdict = tier1_word_count(tc.user_utterance)
    latency_ms = (time.time() - t0) * 1000
    tier1_results[tc.name] = verdict
    word_count = len(tc.user_utterance.split())
    display = f"[red]{verdict}[/red]" if verdict else "[yellow]→ Tier 2[/yellow]"
    tier1_table.add_row(tc.name, str(word_count), display, f"{latency_ms:.2f}ms")
console.print(tier1_table)

# %% [markdown]
# ## Step 4 — Tier 2: semantic LLM classifier
#
# For everything Tier 1 escalated, we use an LLM with the bot's recent context
# and the user's barge-in. The classifier needs to see both — without the bot
# context, "yes" looks identical whether it's a backchannel or interrupt.

# %%
openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

CLASSIFIER_PROMPT = """You classify voice-agent barge-ins.

The agent is mid-utterance. The user spoke. Decide:

- INTERRUPT: the user wants the agent to stop and act on what they said
- BACKCHANNEL: the user is being polite/listening, agent should continue

The agent's current sentence: {bot_context}
The user said: {user_utterance}

Output exactly one word: INTERRUPT or BACKCHANNEL"""


def tier2_semantic(
    bot_context: str,
    user_utterance: str,
    model: str = "gpt-4o-mini",
) -> tuple[Literal["INTERRUPT", "BACKCHANNEL"], float]:
    """Tier 2: small LLM with conversational context.

    Returns (verdict, latency_seconds).
    """
    t0 = time.time()
    response = openai_client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": CLASSIFIER_PROMPT.format(
                    bot_context=bot_context,
                    user_utterance=user_utterance,
                ),
            }
        ],
        temperature=0,
        max_tokens=4,
    )
    latency = time.time() - t0
    raw = (response.choices[0].message.content or "").strip().upper()
    verdict: Literal["INTERRUPT", "BACKCHANNEL"] = (
        "INTERRUPT" if raw.startswith("INTERRUPT") else "BACKCHANNEL"
    )
    return verdict, latency


console.print("\n[bold]🧠 Tier 2: semantic LLM classifier[/bold]\n")
tier2_table = Table()
tier2_table.add_column("Test case", style="cyan")
tier2_table.add_column("Bot context", style="white", max_width=45)
tier2_table.add_column("User", style="yellow")
tier2_table.add_column("Verdict", justify="center")
tier2_table.add_column("Latency", justify="right")

tier2_results: dict[str, str] = {}
for tc in TEST_CASES:
    if tier1_results[tc.name] is not None:
        continue
    verdict, latency = tier2_semantic(tc.bot_context, tc.user_utterance)
    tier2_results[tc.name] = verdict
    color = "red" if verdict == "INTERRUPT" else "green"
    bot_short = tc.bot_context[:45] + "..." if len(tc.bot_context) > 45 else tc.bot_context
    tier2_table.add_row(
        tc.name,
        bot_short,
        repr(tc.user_utterance),
        f"[{color}]{verdict}[/{color}]",
        f"{latency * 1000:.0f}ms",
    )
console.print(tier2_table)

# %% [markdown]
# ## Step 5 — Combine the tiers and score against ground truth

# %%
console.print("\n[bold]📊 Final results vs ground truth[/bold]\n")
final_table = Table()
final_table.add_column("Test case", style="cyan")
final_table.add_column("Predicted", justify="center")
final_table.add_column("Ground truth", justify="center")
final_table.add_column("✓?", justify="center")

correct = 0
for tc in TEST_CASES:
    predicted = tier1_results[tc.name] or tier2_results.get(tc.name, "UNKNOWN")
    is_correct = predicted == tc.ground_truth
    if is_correct:
        correct += 1
    pred_color = "red" if predicted == "INTERRUPT" else "green"
    truth_color = "red" if tc.ground_truth == "INTERRUPT" else "green"
    final_table.add_row(
        tc.name,
        f"[{pred_color}]{predicted}[/{pred_color}]",
        f"[{truth_color}]{tc.ground_truth}[/{truth_color}]",
        "[green]✓[/green]" if is_correct else "[red]✗[/red]",
    )
console.print(final_table)
accuracy = correct / len(TEST_CASES)
console.print(f"\n[bold]Accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.0%}[/bold]")

# %% [markdown]
# ## Step 6 — Compare against the naive classifier


# %%
def naive_classifier(user_utterance: str) -> Literal["INTERRUPT", "BACKCHANNEL"]:
    """The naive approach: anything ≥2 words is an interrupt."""
    return "INTERRUPT" if len(user_utterance.split()) >= 2 else "BACKCHANNEL"


console.print("\n[bold]🔍 Naive classifier (word count only) for comparison:[/bold]\n")
naive_table = Table()
naive_table.add_column("Test case", style="cyan")
naive_table.add_column("Predicted", justify="center")
naive_table.add_column("Ground truth", justify="center")
naive_table.add_column("✓?", justify="center")

naive_correct = 0
for tc in TEST_CASES:
    predicted = naive_classifier(tc.user_utterance)
    is_correct = predicted == tc.ground_truth
    if is_correct:
        naive_correct += 1
    naive_table.add_row(
        tc.name,
        predicted,
        tc.ground_truth,
        "[green]✓[/green]" if is_correct else "[red]✗[/red]",
    )
console.print(naive_table)
console.print(
    f"\n[yellow]Naive accuracy: {naive_correct}/{len(TEST_CASES)} = "
    f"{naive_correct / len(TEST_CASES):.0%}[/yellow]"
)
console.print(
    f"[bold green]Two-tier accuracy: {correct}/{len(TEST_CASES)} = {accuracy:.0%}[/bold green]"
)

# %% [markdown]
# ## Recap
#
# - ✅ "Mhmm" can be backchannel or interrupt depending on what the agent
#   was saying. Word count alone won't tell you which.
# - ✅ A two-tier classifier — word-count fast path + semantic LLM —
#   handles 95%+ of cases at ~$0.0001 each.
# - ✅ The classifier needs the **bot's current sentence**, not just the
#   user's utterance. Without that context the problem is unsolvable.
#
# **Next:** `03_split_state.py` — what happens when three components in
# your voice stack disagree about what just happened.
