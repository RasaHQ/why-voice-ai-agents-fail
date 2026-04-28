"""Shared utilities for the four failure-mode scripts.

This module is intentionally small. It does three things:

1. Loads API credentials from `.env`, the environment, or Colab secrets.
2. Provides a single, friendly Rich console for consistent script output.
3. Provides cheap "is this credential actually valid?" health checks
   that the Makefile can call before running any of the four scripts.

Everything here is pure Python with no I/O at import time. The four
scripts import from `_utils` rather than each maintaining their own
copy of the credential-loading boilerplate.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Final

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
AUDIO_OUTPUT_DIR: Final[Path] = REPO_ROOT / "audio_output"
ENV_PATH: Final[Path] = REPO_ROOT / ".env"

# Load .env once at import. python-dotenv silently does nothing if it
# doesn't exist, which is fine — the user might be supplying env vars
# another way (Colab secrets, CI vars, direnv).
load_dotenv(ENV_PATH, override=False)

# ── Console ───────────────────────────────────────────────────────────────────
console: Final[Console] = Console()


# ── Credential loading ────────────────────────────────────────────────────────
def get_key(name: str, *, required: bool = True) -> str | None:
    """Load an API key from Colab secrets, .env, or the environment.

    Tries (in order):
      1. Google Colab `userdata.get(name)` — no-op outside Colab
      2. `os.environ[name]` — covers .env (already loaded above) and CI

    If `required=True` and nothing is found, raises RuntimeError.
    Returns None when `required=False` and nothing is found.
    """
    key: str | None = None
    try:
        from google.colab import userdata  # type: ignore[import-not-found]

        key = userdata.get(name)
    except (ImportError, Exception):
        # Not in Colab, or the secret isn't set there. Fall through.
        pass

    if not key:
        key = os.environ.get(name)

    if required and not key:
        raise RuntimeError(
            f"{name} is not set. Add it to .env, export it, "
            f"or set it as a Colab secret. See .env.example."
        )
    return key


def ensure_audio_dir() -> Path:
    """Make sure the audio output directory exists and return its path."""
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return AUDIO_OUTPUT_DIR


# ── Health checks ─────────────────────────────────────────────────────────────
# These are deliberately cheap. Each one makes the smallest possible
# request that proves the key is valid. We don't synthesize audio or
# burn LLM tokens — that's what the actual scripts are for.


def check_deepgram() -> bool:
    """Verify DEEPGRAM_API_KEY by hitting the /v1/projects endpoint."""
    key = get_key("DEEPGRAM_API_KEY", required=False)
    if not key:
        console.print("[red]✗ DEEPGRAM_API_KEY not set[/red]")
        return False
    try:
        r = requests.get(
            "https://api.deepgram.com/v1/projects",
            headers={"Authorization": f"Token {key}"},
            timeout=10,
        )
        if r.status_code == 200:
            console.print("[green]✓ Deepgram key valid[/green]")
            return True
        console.print(f"[red]✗ Deepgram returned HTTP {r.status_code}[/red]")
        return False
    except requests.RequestException as e:
        console.print(f"[red]✗ Deepgram check failed: {e}[/red]")
        return False


def check_rime() -> bool:
    """Verify RIME_API_KEY with a 1-character TTS request (cheapest legal call)."""
    key = get_key("RIME_API_KEY", required=False)
    if not key:
        console.print("[red]✗ RIME_API_KEY not set[/red]")
        return False
    try:
        r = requests.post(
            "https://users.rime.ai/v1/rime-tts",
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "audio/wav",
                "Content-Type": "application/json",
            },
            json={
                "speaker": "abbie",
                "text": ".",
                "modelId": "mistv2",
                "samplingRate": 16000,
            },
            timeout=15,
        )
        if r.status_code == 200 and r.content:
            console.print("[green]✓ Rime key valid[/green]")
            return True
        console.print(f"[red]✗ Rime returned HTTP {r.status_code}[/red]")
        return False
    except requests.RequestException as e:
        console.print(f"[red]✗ Rime check failed: {e}[/red]")
        return False


def check_openai() -> bool:
    """Verify OPENAI_API_KEY by listing models (no token cost)."""
    key = get_key("OPENAI_API_KEY", required=False)
    if not key:
        console.print("[red]✗ OPENAI_API_KEY not set[/red]")
        return False
    try:
        import openai

        client = openai.OpenAI(api_key=key)
        # `.list()` is GET /v1/models, returns immediately, no token cost.
        models = client.models.list()
        # Just access the first model name to force the iterator.
        next(iter(models), None)
        console.print("[green]✓ OpenAI key valid[/green]")
        return True
    except Exception as e:
        console.print(f"[red]✗ OpenAI check failed: {e}[/red]")
        return False


def check_all() -> None:
    """Run all three checks and print a summary table.

    Exits with code 1 if any check fails — this is what the Makefile
    relies on for `keys-check` to gate `make scripts`.
    """
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="white")
    table.add_column("Status", justify="center")

    results = {
        "Deepgram": check_deepgram(),
        "Rime": check_rime(),
        "OpenAI": check_openai(),
    }

    console.print()
    for provider, ok in results.items():
        table.add_row(provider, "[green]OK[/green]" if ok else "[red]FAIL[/red]")
    console.print(table)

    if not all(results.values()):
        console.print(
            "\n[yellow]⚠ Some credentials are missing or invalid. "
            "Copy .env.example to .env and fill it in.[/yellow]"
        )
        sys.exit(1)
    console.print("\n[green]All credentials OK — ready to run the scripts.[/green]")
