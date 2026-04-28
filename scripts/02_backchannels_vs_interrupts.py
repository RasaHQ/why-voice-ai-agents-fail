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
# When a user says "mhmm" while the agent is talking, are they being polite
# (keep going) or interrupting (stop and listen)? The signal is in *what
# the agent was just saying*, not in the user's word.
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
from typing import Literal

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
    "Failure 02 — Tools that look like backchannels",
    "Same syllable, opposite intent depending on what the agent was saying.",
)

narrate(
    "I'm going to play six barge-ins. Three are 'keep going' and three are "
    "'stop and act on this'. The trick: you can't tell from the user's word "
    "alone — you need the bot's last sentence.",
    style="italic dim",
)

# %% [markdown]
# ## Step 1 — Set up six real conversation moments


# %%
@dataclass
class TestCase:
    """One barge-in moment: bot context + user utterance + ground truth label."""

    name: str
    bot_context: str
    user_utterance: str
    ground_truth: Literal["INTERRUPT", "BACKCHANNEL"]
    description: str


TEST_CASES: list[TestCase] = [
    TestCase(
        name="polite_listening",
        bot_context="Your November bill came to four hundred and eighty-two dollars, which is up from",
        user_utterance="Mhmm.",
        ground_truth="BACKCHANNEL",
        description="Bot is reading numbers. User signals attention.",
    ),
    TestCase(
        name="answering_question",
        bot_context="Are you inquiring about your current account, or a different one?",
        user_utterance="Yes.",
        ground_truth="INTERRUPT",
        description="Bot asked a yes/no question. User answered.",
    ),
    TestCase(
        name="filled_pause_during_explanation",
        bot_context="The increase comes from a one-time charge applied during the billing cycle",
        user_utterance="Right.",
        ground_truth="BACKCHANNEL",
        description="Bot is explaining. User signals understanding.",
    ),
    TestCase(
        name="genuine_stop_request",
        bot_context="I can transfer your call to a specialist who handles billing disputes,",
        user_utterance="Wait, hold on.",
        ground_truth="INTERRUPT",
        description="Bot is offering. User wants to pause the action.",
    ),
    TestCase(
        name="information_correction",
        bot_context="So I'll process the refund to your account ending in three four five six,",
        user_utterance="No, that's not right.",
        ground_truth="INTERRUPT",
        description="Bot is about to act. User stops it.",
    ),
    TestCase(
        name="agreement_filler",
        bot_context="And then we'll send you a confirmation email within twenty-four hours,",
        user_utterance="Okay.",
        ground_truth="BACKCHANNEL",
        description="Bot is wrapping up a sentence. User says 'okay, go on'.",
    ),
]

step(1, "Synthesize the six barge-ins with Rime")
for tc in TEST_CASES:
    audio_path = AUDIO_DIR / f"02_bargein_{tc.name}.mp3"
    if not audio_path.exists():
        with live_status(f"Synthesizing: '{tc.user_utterance}'"):
            speed = 1.15 if tc.ground_truth == "BACKCHANNEL" else 1.0
            synthesize(tc.user_utterance, audio_path, speed_alpha=speed, audio_format="mp3")
    console.print(
        f"  [green]✓[/green] [yellow]{tc.user_utterance!r:<28}[/yellow] [dim]→ {audio_path.name}[/dim]"
    )

# %% [markdown]
# ## Step 2 — Listen to a few. Same word, different meaning.

# %%
section("Listen — same syllable, different intent")

# Play three contrasting pairs so the audience hears the ambiguity in their ears.
preview_pairs = [
    ("polite_listening", "answering_question"),
    ("filled_pause_during_explanation", "agreement_filler"),
]

for tc in TEST_CASES:
    if tc.name not in [name for pair in preview_pairs for name in pair][:4]:
        continue
    color = "green" if tc.ground_truth == "BACKCHANNEL" else "red"
    console.print()
    console.print(f"  [bold]🤖 Agent:[/bold] [italic]{tc.bot_context}...[/italic]")
    console.print(f"  [bold]👤 User barges in:[/bold] [yellow]{tc.user_utterance!r}[/yellow]")
    audio_path = AUDIO_DIR / f"02_bargein_{tc.name}.mp3"
    play(audio_path, label=f"Listen — {tc.description}")
    console.print(f"  [bold]Truth:[/bold] [{color}]{tc.ground_truth}[/{color}]")
    pause_for_effect(0.4)

# %% [markdown]
# ## Step 3 — Tier 1 classifier: word-count fast path

# %%
section("Tier 1 — word-count fast path (zero-cost, zero-latency)")

narrate(
    "Sustained speech is never a backchannel. If the user has been talking "
    "for more than ~4 words, it's an interrupt. No human says 'okay yes I "
    "really agree with that' as a backchannel.",
)


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


tier1_table = Table(title="Tier 1 verdicts", show_header=True)
tier1_table.add_column("User said", style="yellow", max_width=22)
tier1_table.add_column("Words", justify="right", style="cyan")
tier1_table.add_column("Verdict", justify="center")
tier1_table.add_column("Latency", justify="right", style="dim")

tier1_results: dict[str, str | None] = {}
for tc in TEST_CASES:
    t0 = time.time()
    verdict = tier1_word_count(tc.user_utterance)
    latency_ms = (time.time() - t0) * 1000
    tier1_results[tc.name] = verdict
    word_count = len(tc.user_utterance.split())
    display = f"[red]{verdict}[/red]" if verdict else "[yellow]→ Tier 2[/yellow]"
    tier1_table.add_row(repr(tc.user_utterance), str(word_count), display, f"{latency_ms:.2f}ms")
console.print(tier1_table)

# %% [markdown]
# ## Step 4 — Tier 2 classifier: small LLM with bot context

# %%
section("Tier 2 — small LLM with the bot's last sentence")

narrate(
    "For everything Tier 1 escalated, we ask a small LLM with the bot's "
    "last sentence and the user's words. The model's job: pick INTERRUPT "
    "or BACKCHANNEL.",
)

llm_client = make_llm_client()

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
    model: str = DEFAULT_NEBIUS_MODEL,
) -> tuple[Literal["INTERRUPT", "BACKCHANNEL"], float]:
    """Tier 2: small LLM with conversational context.

    Returns (verdict, latency_seconds).
    """
    t0 = time.time()
    response = llm_client.chat.completions.create(
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


tier2_table = Table(title="Tier 2 verdicts", show_header=True)
tier2_table.add_column("User said", style="yellow", max_width=22)
tier2_table.add_column("Bot was saying", style="white", max_width=50)
tier2_table.add_column("Verdict", justify="center")
tier2_table.add_column("Latency", justify="right", style="dim")

tier2_results: dict[str, str] = {}
total_latency = 0.0
total_calls = 0

for tc in TEST_CASES:
    if tier1_results[tc.name] is not None:
        continue
    with live_status(f"Classifying: '{tc.user_utterance}'"):
        verdict, latency = tier2_semantic(tc.bot_context, tc.user_utterance)
    tier2_results[tc.name] = verdict
    color = "red" if verdict == "INTERRUPT" else "green"
    bot_short = tc.bot_context[:50] + "..." if len(tc.bot_context) > 50 else tc.bot_context
    tier2_table.add_row(
        repr(tc.user_utterance),
        bot_short,
        f"[{color}]{verdict}[/{color}]",
        f"{latency * 1000:.0f}ms",
    )
    total_latency += latency
    total_calls += 1

console.print(tier2_table)

if total_calls:
    avg_ms = (total_latency / total_calls) * 1000
    console.print(
        f"  [dim]Average Tier-2 latency: {avg_ms:.0f}ms — "
        f"this is the cost you pay for short barge-ins.[/dim]"
    )

# %% [markdown]
# ## Step 5 — Score against ground truth + naive baseline

# %%
section("Final results vs ground truth")

final_table = Table(title="Two-tier classifier", show_header=True)
final_table.add_column("Test case", style="cyan")
final_table.add_column("User said", style="yellow")
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
        repr(tc.user_utterance),
        f"[{pred_color}]{predicted}[/{pred_color}]",
        f"[{truth_color}]{tc.ground_truth}[/{truth_color}]",
        "[green]✓[/green]" if is_correct else "[red]✗[/red]",
    )
console.print(final_table)
two_tier_acc = correct / len(TEST_CASES)


def naive_classifier(user_utterance: str) -> Literal["INTERRUPT", "BACKCHANNEL"]:
    """The naive approach: anything ≥2 words is an interrupt."""
    return "INTERRUPT" if len(user_utterance.split()) >= 2 else "BACKCHANNEL"


naive_table = Table(title="Naive classifier (word-count only)", show_header=True)
naive_table.add_column("User said", style="yellow")
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
        repr(tc.user_utterance),
        predicted,
        tc.ground_truth,
        "[green]✓[/green]" if is_correct else "[red]✗[/red]",
    )
console.print()
console.print(naive_table)
naive_acc = naive_correct / len(TEST_CASES)

# %% [markdown]
# ### Punchline

# %%
delta = (two_tier_acc - naive_acc) * 100
if two_tier_acc >= naive_acc:
    punchline(
        f"Two-tier: {correct}/{len(TEST_CASES)} = {two_tier_acc:.0%}\n"
        f"Naive:    {naive_correct}/{len(TEST_CASES)} = {naive_acc:.0%}\n"
        f"Delta:    +{delta:.0f}pp on the cases that matter — "
        f"the short, ambiguous ones.",
        kind="win",
    )
else:
    punchline(
        f"Two-tier underperformed naive on this run "
        f"({correct}/{len(TEST_CASES)} vs {naive_correct}/{len(TEST_CASES)}).\n"
        f"Try a different model with NEBIUS_MODEL, or check the prompt.",
        kind="info",
    )

# %% [markdown]
# ## The takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]The signal isn't in the user's word — it's in the relationship[/bold]\n"
    "  [bold]between what the bot was saying and what the user said.[/bold]\n"
)
console.print(
    "  [dim]Word count alone misses the cases where it costs the most:[/dim]\n"
    "  [dim]'Yes' to a yes/no question is an interrupt. Identical 'yes' during a statement is filler.[/dim]\n"
)

console.print()
console.print(
    "[bold magenta]Next:[/bold magenta]  make script-03   [dim]— The system disagrees with itself[/dim]"
)
console.print()
