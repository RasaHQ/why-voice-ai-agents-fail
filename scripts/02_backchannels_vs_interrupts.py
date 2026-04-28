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
# Run from the repo root: `make script-02`

# %%
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

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
    step,
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
    "Failure 02 — Tools that look like backchannels",
    "Same syllable. Opposite meaning. The bot's last sentence is the only thing that disambiguates.",
)

# %% [markdown]
# ## What we're about to do

# %%
watch_this(
    "I'll play the EXACT SAME WORD — 'Yes' — twice.\n"
    "First it's the user being polite (keep going).\n"
    "Second it's the user answering a question (stop and act).\n"
    "A naive 'is the user talking?' classifier gets one of them wrong.\n"
    "The fix needs the BOT'S last sentence as context.",
)


# %% [markdown]
# ## Step 1 — Two clips. Identical word. Opposite intent.


# %%
@dataclass
class TestCase:
    """One barge-in moment: bot context + user utterance + ground truth label."""

    name: str
    bot_context: str
    user_utterance: str
    ground_truth: Literal["INTERRUPT", "BACKCHANNEL"]
    setup: str  # what the speaker should say before playing this clip


# Hero pair — these are the ones we play out loud. Identical user word,
# opposite truth, only the bot context tells us which.
HERO_PAIR: list[TestCase] = [
    TestCase(
        name="yes_during_statement",
        bot_context="Your November bill came to four hundred and eighty-two dollars, which is up from",
        user_utterance="Yes.",
        ground_truth="BACKCHANNEL",
        setup="Bot is reading numbers. User says 'yes' = 'go on'.",
    ),
    TestCase(
        name="yes_to_question",
        bot_context="Are you inquiring about your current account, or a different one?",
        user_utterance="Yes.",
        ground_truth="INTERRUPT",
        setup="Bot asked a yes/no question. User says 'yes' = 'stop and act'.",
    ),
]

# Wider test set — these run through the classifier silently, just for the
# accuracy comparison at the end.
EXTRA_CASES: list[TestCase] = [
    TestCase(
        name="mhmm_polite",
        bot_context="The increase comes from a one-time charge applied during the billing cycle",
        user_utterance="Mhmm.",
        ground_truth="BACKCHANNEL",
        setup="",
    ),
    TestCase(
        name="genuine_stop",
        bot_context="I can transfer your call to a specialist who handles billing disputes,",
        user_utterance="Wait, hold on.",
        ground_truth="INTERRUPT",
        setup="",
    ),
    TestCase(
        name="information_correction",
        bot_context="So I'll process the refund to your account ending in three four five six,",
        user_utterance="No, that's not right.",
        ground_truth="INTERRUPT",
        setup="",
    ),
    TestCase(
        name="okay_filler",
        bot_context="And then we'll send you a confirmation email within twenty-four hours,",
        user_utterance="Okay.",
        ground_truth="BACKCHANNEL",
        setup="",
    ),
]

ALL_CASES = HERO_PAIR + EXTRA_CASES

step(1, "Synthesize the two hero clips with Rime")
for tc in HERO_PAIR:
    audio_path = AUDIO_DIR / f"02_{tc.name}.mp3"
    if not audio_path.exists():
        with live_status(f"Synthesizing '{tc.user_utterance}'"):
            synthesize(tc.user_utterance, audio_path, audio_format="mp3")
    console.print(
        f"  [green]✓[/green] [yellow]{tc.user_utterance!r}[/yellow] [dim]({audio_path.name})[/dim]"
    )

# Also synthesize the extras silently (used later by the classifier).
for tc in EXTRA_CASES:
    audio_path = AUDIO_DIR / f"02_{tc.name}.mp3"
    if not audio_path.exists():
        synthesize(tc.user_utterance, audio_path, audio_format="mp3")

# %% [markdown]
# ## Step 2 — Listen. Same word. Opposite meaning.

# %%
section("Listen — same word, opposite meaning")

for tc in HERO_PAIR:
    color = "green" if tc.ground_truth == "BACKCHANNEL" else "red"
    label = "BACKCHANNEL (keep going)" if tc.ground_truth == "BACKCHANNEL" else "INTERRUPT (stop)"
    console.print()
    console.print(
        f"  [bold magenta]🤖 Agent says:[/bold magenta] [italic]{tc.bot_context}...[/italic]"
    )
    console.print(
        f"  [bold cyan]👤 User says:[/bold cyan]   [bold yellow]{tc.user_utterance!r}[/bold yellow]"
    )
    audio_path = AUDIO_DIR / f"02_{tc.name}.mp3"
    play(audio_path, label="(playing)")
    console.print(f"  [bold]Truth:[/bold] [bold {color}]{label}[/bold {color}]")
    pause_for_effect(0.5)

# %% [markdown]
# ## Step 3 — A naive classifier looks at the user's word. It's wrong.

# %%
section("First try — the naive 'word count' classifier")

narrate(
    "First attempt at solving this: 'short user utterance = backchannel, "
    "long = interrupt'. Sounds reasonable. Let's see how it does."
)


def naive_classifier(user_utterance: str) -> Literal["INTERRUPT", "BACKCHANNEL"]:
    """The naive approach: anything ≥2 words is an interrupt."""
    return "INTERRUPT" if len(user_utterance.split()) >= 2 else "BACKCHANNEL"


# Show the naive verdict on the two hero cases — that's the key contrast.
naive_correct = 0
for tc in HERO_PAIR:
    pred = naive_classifier(tc.user_utterance)
    correct = pred == tc.ground_truth
    if correct:
        naive_correct += 1
    color = "green" if correct else "red"
    mark = "✓" if correct else "✗ WRONG"
    console.print(
        f"  [yellow]{tc.user_utterance!r}[/yellow] "
        f"→ predicted [bold]{pred}[/bold]  "
        f"[bold {color}]{mark}[/bold {color}]"
    )

# Quietly score on the extras too for the final tally.
naive_total_correct = naive_correct + sum(
    1 for tc in EXTRA_CASES if naive_classifier(tc.user_utterance) == tc.ground_truth
)

verdict(
    "The naive classifier gets the IDENTICAL word wrong half the time.",
    "It can't possibly tell 'yes-keep-going' apart from 'yes-stop-and-act' "
    "because it never looks at what the bot was saying.",
    kind="fail",
)

# %% [markdown]
# ## Step 4 — The fix: include the bot's last sentence.

# %%
section("Now the fix — add the bot's last sentence as context")

narrate(
    "Two-tier classifier. Tier 1: if user said >4 words, it's always an interrupt "
    "(humans don't backchannel for that long). Tier 2: small LLM with the bot's "
    "last sentence + the user's words → INTERRUPT or BACKCHANNEL."
)

llm_client = make_llm_client()

CLASSIFIER_PROMPT = """You classify voice-agent barge-ins.

The agent is mid-utterance. The user spoke. Decide:

- INTERRUPT: the user wants the agent to stop and act on what they said
- BACKCHANNEL: the user is being polite/listening, agent should continue

The agent's current sentence: {bot_context}
The user said: {user_utterance}

Output exactly one word: INTERRUPT or BACKCHANNEL"""


def two_tier_classify(
    bot_context: str, user_utterance: str
) -> tuple[Literal["INTERRUPT", "BACKCHANNEL"], float]:
    """Tier 1 + Tier 2."""
    if len(user_utterance.split()) > 4:
        return "INTERRUPT", 0.0  # tier 1 short-circuit, no LLM cost

    t0 = time.time()
    response = llm_client.chat.completions.create(
        model=DEFAULT_NEBIUS_MODEL,
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
    pred: Literal["INTERRUPT", "BACKCHANNEL"] = (
        "INTERRUPT" if raw.startswith("INTERRUPT") else "BACKCHANNEL"
    )
    return pred, latency


# Show the fix on the two hero cases first.
fix_correct = 0
for tc in HERO_PAIR:
    with live_status(f"Two-tier classifier on '{tc.user_utterance}'"):
        pred, latency = two_tier_classify(tc.bot_context, tc.user_utterance)
    correct = pred == tc.ground_truth
    if correct:
        fix_correct += 1
    color = "green" if correct else "red"
    mark = "✓" if correct else "✗ WRONG"
    latency_str = f"{latency * 1000:.0f}ms" if latency > 0 else "<1ms (Tier 1)"
    console.print(
        f"  [yellow]{tc.user_utterance!r}[/yellow] + bot context "
        f"→ predicted [bold]{pred}[/bold]  "
        f"[bold {color}]{mark}[/bold {color}]  [dim]{latency_str}[/dim]"
    )

# Score on the extras too.
fix_total_correct = fix_correct
for tc in EXTRA_CASES:
    pred, _ = two_tier_classify(tc.bot_context, tc.user_utterance)
    if pred == tc.ground_truth:
        fix_total_correct += 1

# %% [markdown]
# ## Big comparison

# %%
total = len(ALL_CASES)
big_compare(
    "Naive (word count only)",
    f"{naive_total_correct} / {total}\n\n({int(100 * naive_total_correct / total)}% accuracy)",
    "Two-tier (with bot context)",
    f"{fix_total_correct} / {total}\n\n({int(100 * fix_total_correct / total)}% accuracy)",
)

if fix_total_correct > naive_total_correct:
    delta = fix_total_correct - naive_total_correct
    verdict(
        f"Two-tier classifier got {delta} more case(s) right.",
        "It's a small absolute number — but those are the cases where it costs the most. "
        "Saying 'yes' to a yes/no question is a customer answering you. "
        "Cut them off and they hang up.",
        kind="win",
    )
else:
    verdict(
        f"Two-tier matched the naive baseline this run "
        f"({fix_total_correct}/{total} vs {naive_total_correct}/{total}).",
        "Try a different model with NEBIUS_MODEL, or add more ambiguous test cases.",
        kind="info",
    )

# %% [markdown]
# ## Takeaway

# %%
section("Takeaway")
console.print(
    "  [bold]The signal isn't in the user's word.[/bold]\n"
    "  [bold cyan]It's in the relationship between what the bot said and what the user said.[/bold cyan]\n"
)

console.print()
console.print(
    "[bold magenta]Next:[/bold magenta]  [green]make script-03[/green]   [dim]— The system disagrees with itself[/dim]"
)
console.print()
