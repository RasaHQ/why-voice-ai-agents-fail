# Why (Voice) AI Agents Fail

> Companion code for the talk **"Why (Voice) AI Agents Fail"** by Rod Rivera (Rasa).

Four runnable Python scripts demonstrating the four most common failure modes of production voice agents in 2026 — and the architectural fix for each.

Every script uses **real Deepgram ASR** and **real Rime TTS**. They run end-to-end on your machine in under a minute each.

---

## What's in here

| # | Script | Failure mode | What it shows |
|---|--------|--------------|---------------|
| 01 | [`scripts/01_turn_taking.py`](scripts/01_turn_taking.py) | Turn-taking is harder than you think | ASR endpoints on acoustic pauses, not semantic completion. We synthesize a phone number with a natural mid-utterance pause, watch Deepgram split it, then add a semantic turn-taking layer that fixes it. |
| 02 | [`scripts/02_backchannels_vs_interrupts.py`](scripts/02_backchannels_vs_interrupts.py) | Tools that look like backchannels | "Mhmm" can mean "keep going" or "stop and listen" depending on what the agent was saying. We build a two-tier classifier (word count + small LLM) and measure it against ground truth. |
| 03 | [`scripts/03_split_state.py`](scripts/03_split_state.py) | The system disagrees with itself | Voice agents are concurrent. While the agent is mid-tool-call, the user can barge in. Without cancellation, the agent finishes work the user already canceled. We show the failure and the cooperative-cancellation fix. |
| 04 | [`scripts/04_premature_goodbye.py`](scripts/04_premature_goodbye.py) | The bot says goodbye too soon | Pure-agentic architectures end calls when the user says "alright that makes sense." We compare a pure-agentic agent against one with **progressive control** — agentic by default, deterministic guardrails at the seams that matter. |

---

## Quickstart

You need Python 3.11+ and [`uv`](https://docs.astral.sh/uv/) installed. If you don't have `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone, set up, and run:

```bash
git clone https://github.com/RasaHQ/why-voice-ai-agents-fail.git
cd why-voice-ai-agents-fail

# Create the virtual environment and install dependencies
make install

# Copy the env template and fill in your API keys
cp .env.example .env
# Edit .env with DEEPGRAM_API_KEY, RIME_API_KEY, OPENAI_API_KEY

# Verify the environment and credentials
make check-env

# Run all four scripts in sequence
make scripts
```

Or run them one at a time:

```bash
make script-01    # Failure 1: Turn-taking
make script-02    # Failure 2: Backchannels vs interrupts
make script-03    # Failure 3: Split state
make script-04    # Failure 4: Premature goodbye
```

Each script runs in 30–60 seconds and costs under a cent in API calls.

---

## Prerequisites

You need three API keys. Free tiers cover the entire script series.

| Provider | What it's used for | Sign up |
|----------|-------------------|---------|
| **Deepgram** | Streaming ASR (speech-to-text) | https://deepgram.com |
| **Rime** | TTS synthesis with controllable pauses | https://rime.ai |
| **OpenAI** | Small classifiers and the agent loop in scripts 1, 2, 3, 4 | https://platform.openai.com |

Set them in a `.env` file at the repo root, or export them as environment variables. The scripts also auto-detect Google Colab Secrets if you'd rather run them in a notebook.

To verify your credentials before running anything:

```bash
make keys-check
```

---

## How the scripts are structured

Each script follows the same pedagogical pattern:

1. **Show the failure first.** We synthesize or simulate the bad behavior so you can hear or read the actual failure mode in the agent's own voice.
2. **Diagnose what went wrong.** Each script explains why the failure happens architecturally — usually a mismatch between what the model is being asked to do and what the engineer thinks it's doing.
3. **Apply the fix.** We add the architectural pattern that addresses the failure and re-run the same input. The fix isn't a smarter model. It's an engineering change in the harness.

The scripts are written in [jupytext](https://jupytext.readthedocs.io/) percent format, so they convert cleanly to Jupyter notebooks if you'd prefer that workflow:

```bash
make notebooks    # writes notebooks/01_turn_taking.ipynb etc.
```

---

## Make targets

The full list (also available via `make help`):

### Setup
- `make env` — Create the `.venv` with `uv`
- `make install` — Install runtime + dev dependencies
- `make lock` — Generate `uv.lock` for reproducible builds
- `make sync` — Sync env to match `uv.lock`
- `make check-env` — Verify Python, uv, and dependencies

### Run
- `make script-01` … `make script-04` — Run a single failure-mode script
- `make scripts` — Run all four in sequence (after `keys-check`)

### Credential checks
- `make deepgram-check` — Verify `DEEPGRAM_API_KEY` is valid
- `make rime-check` — Verify `RIME_API_KEY` is valid
- `make openai-check` — Verify `OPENAI_API_KEY` is valid
- `make keys-check` — All three at once, with a summary table

### Quality
- `make format` — Auto-format with `ruff`
- `make lint` — Lint with `ruff` + `mypy`
- `make test` — Run the test suite
- `make check` — `format` + `lint` + `test`

### Housekeeping
- `make clean` — Remove caches and build artefacts
- `make clean-output` — Remove generated audio files
- `make clean-all` — Both, but keep the venv
- `make nuke` — Clean everything, including the venv
- `make structure` — Show the project tree

---

## Project layout

```
why-voice-ai-agents-fail/
├── scripts/
│   ├── _utils.py                          # Shared helpers (credentials, console, checks)
│   ├── 01_turn_taking.py
│   ├── 02_backchannels_vs_interrupts.py
│   ├── 03_split_state.py
│   └── 04_premature_goodbye.py
├── tests/
│   └── test_utils.py                      # Smoke tests for the credential helpers
├── audio_output/                          # Generated by the scripts (git-ignored)
├── .env.example
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

---

## What's deliberately NOT in this repo

The companion talk references several adjacent topics that we kept out of the script set to keep them focused on demonstrable failures:

- **The slide deck.** Slides for the talk live in a separate internal repo. This repo is the public-facing artefact for engineers who want to reproduce the failures and the fixes on their own machines.
- **Sentiment-aware response generation.** We've experimented with this in production. The latency overhead is real and the quality lift is small enough that we can't yet recommend it for general use.
- **Multi-tier filler architectures.** Genuinely useful, but it's an optimization rather than a failure mode.
- **Multilingual turn-taking.** Backchannel patterns and pause durations vary by language. The scripts here are English-only.

These are covered in the broader course **Why Agents Fail**, launching as part of Rasa University. → **rasa.com/why-agents-fail**

---

## Contributing

Found a bug? Open an issue. Have a better example or a clearer explanation? Open a PR. The course is a living document and the repo is the same.

---

## License

MIT for code. CC BY-SA 4.0 for the documentation and script narrative.

---

*Why (Voice) AI Agents Fail — a Rasa course taught by Rod Rivera*
