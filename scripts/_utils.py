"""Shared utilities for the four failure-mode scripts.

This module holds the things every script needs so the failure-mode files
themselves stay focused on what they're demonstrating, not on plumbing.

Specifically:

1. Loads API credentials from `.env`, the environment, or Colab secrets.
2. Provides a single, friendly Rich console for consistent script output.
3. Provides a shared `make_llm_client()` factory that returns a Nebius-backed
   OpenAI client — with sensible timeouts and retries.
4. Provides `synthesize()` and `play()` helpers so every script speaks audio
   the same way and plays it through the host's speakers.
5. Provides narrative helpers (`narrate()`, `pause_for_effect()`, `live_status()`)
   so the scripts feel like a presentation, not a wall of tables.
6. Provides cheap "is this credential actually valid?" health checks
   that the Makefile and `verify_setup.py` call before running the scripts.

Why MP3 and not WAV? Rime's WAV output from `mistv2` doesn't always carry the
full RIFF header the way Deepgram expects, which surfaces as a 400 "corrupt
or unsupported data" error. MP3 has well-defined sync words and Deepgram
ingests it natively. We use MP3 throughout.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Final

import openai
import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
AUDIO_OUTPUT_DIR: Final[Path] = REPO_ROOT / "audio_output"
ENV_PATH: Final[Path] = REPO_ROOT / ".env"

load_dotenv(ENV_PATH, override=False)

# ── Nebius defaults ───────────────────────────────────────────────────────────
NEBIUS_BASE_URL: Final[str] = os.environ.get(
    "NEBIUS_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
)

# Default model. Llama 3.1 8B Instruct is fast enough for the classifier and
# short-reply work the four scripts do, and is currently available on Nebius
# Token Factory. Override via NEBIUS_MODEL.
DEFAULT_NEBIUS_MODEL: Final[str] = os.environ.get(
    "NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

# Audio playback can be disabled (CI, headless, "I just want the data").
# Default is on — this is a voice demo.
PLAY_AUDIO: Final[bool] = os.environ.get("PLAY_AUDIO", "true").lower() != "false"

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


# ── LLM client factory ────────────────────────────────────────────────────────
def make_llm_client(timeout: float = 60.0, max_retries: int = 2) -> openai.OpenAI:
    """Return an OpenAI SDK client wired up to Nebius Token Factory.

    The OpenAI SDK is the client library; the `base_url` points at Nebius's
    OpenAI-compatible endpoint.

    Defaults to a 60-second timeout (Nebius `-fast` variants reply in ~1s,
    full models in ~3-5s; 60s is enough margin for cold starts). Two retries
    with backoff on transient errors.
    """
    api_key = get_key("NEBIUS_API_KEY")
    return openai.OpenAI(
        api_key=api_key,
        base_url=NEBIUS_BASE_URL,
        timeout=timeout,
        max_retries=max_retries,
    )


# ── TTS via Rime ──────────────────────────────────────────────────────────────
# We default to MP3 because Deepgram's auto-detection is more reliable on it,
# and every desktop OS plays MP3 out of the box.

RIME_ENDPOINT: Final[str] = "https://users.rime.ai/v1/rime-tts"


def synthesize(
    text: str,
    output_path: Path,
    *,
    speaker: str = "abbie",
    model_id: str = "mistv2",
    speed_alpha: float = 1.0,
    audio_format: str = "mp3",
) -> Path:
    """Generate audio from text using Rime TTS.

    Defaults to MP3 because that's what works most reliably across the
    Rime → Deepgram → host-speaker pipeline. WAV is also supported by Rime
    but the headers can confuse Deepgram with `mistv2`.

    Returns the output path on success.
    """
    rime_key = get_key("RIME_API_KEY")
    accept = {"mp3": "audio/mp3", "wav": "audio/wav"}.get(audio_format, "audio/mp3")

    response = requests.post(
        RIME_ENDPOINT,
        headers={
            "Authorization": f"Bearer {rime_key}",
            "Accept": accept,
            "Content-Type": "application/json",
        },
        json={
            "speaker": speaker,
            "text": text,
            "modelId": model_id,
            "speedAlpha": speed_alpha,
            # Rime's `samplingRate` only applies to WAV/PCM. For MP3 the
            # encoder picks a sensible default.
            **({"samplingRate": 22050} if audio_format == "wav" else {}),
        },
        timeout=30,
    )
    response.raise_for_status()

    # Sanity-check: did we get audio? Rime occasionally returns an error JSON
    # body with `Content-Type: application/json` even on `Accept: audio/*`.
    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type or len(response.content) < 256:
        raise RuntimeError(
            f"Rime returned non-audio response (Content-Type={content_type!r}, "
            f"size={len(response.content)} bytes): {response.text[:200]}"
        )

    output_path.write_bytes(response.content)
    return output_path


# ── Audio playback ────────────────────────────────────────────────────────────
# Cross-platform playback by shelling out to a system player. Each platform
# has at least one installed by default. We deliberately do NOT take a hard
# dependency on PyAudio / simpleaudio / sounddevice — those need C compilation
# and portaudio installed at the OS level, which is a big ask for a demo repo.

_MAC_PLAYER = "afplay"
_LINUX_PLAYERS = ["mpg123", "ffplay", "play", "paplay", "aplay"]
_WIN_FALLBACK = "cmd.exe"


def _find_linux_player() -> list[str] | None:
    """Return the argv list for a Linux audio player that's actually installed."""
    for cmd in _LINUX_PLAYERS:
        if shutil.which(cmd):
            if cmd == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if cmd == "play":
                return ["play", "-q"]
            return [cmd]
    return None


def play(path: Path, *, label: str | None = None) -> None:
    """Play an audio file through the host's default speakers.

    Honors the PLAY_AUDIO env var — set `PLAY_AUDIO=false` to skip playback
    (useful in CI). Always prints a status line so the user knows what would
    have played even if playback is off.
    """
    if label:
        console.print(f"  [magenta]🔊 {label}[/magenta]  [dim]({path.name})[/dim]")

    if not PLAY_AUDIO:
        console.print("  [dim]   (playback disabled — set PLAY_AUDIO=true to hear it)[/dim]")
        return

    system = platform.system()
    argv: list[str] | None = None

    if system == "Darwin":
        argv = [_MAC_PLAYER, str(path)] if shutil.which(_MAC_PLAYER) else None
    elif system == "Linux":
        prefix = _find_linux_player()
        argv = [*prefix, str(path)] if prefix else None
    elif system == "Windows":
        # `start /wait` blocks until the file finishes playing in Windows.
        argv = [_WIN_FALLBACK, "/c", "start", "/wait", "", str(path)]

    if argv is None:
        console.print(
            "  [yellow]⚠ No audio player found.[/yellow] "
            "[dim]Install: brew install mpg123 (mac), apt install mpg123 (linux).[/dim]"
        )
        return

    try:
        subprocess.run(
            argv,
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError) as e:
        console.print(f"  [yellow]⚠ Audio playback failed: {e}[/yellow]")


# ── Narrative helpers ─────────────────────────────────────────────────────────
# These exist because a voice talk needs more than tables. The scripts use
# these to set up each demo with a sentence or two of context, then deliver
# the punchline after the demo runs.


def header(title: str, subtitle: str = "") -> None:
    """Open a script with a big banner panel."""
    body = Text(title, style="bold white")
    if subtitle:
        body.append("\n" + subtitle, style="dim italic")
    console.print()
    console.print(Panel(body, border_style="cyan", padding=(1, 4)))
    console.print()


def section(title: str) -> None:
    """Mark a major section in a script."""
    console.print()
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")
    console.print()


def narrate(text: str, *, style: str = "white") -> None:
    """Print a narrative paragraph — the 'storytelling' between demos."""
    console.print(f"  [{style}]{text}[/{style}]")


def punchline(text: str, *, kind: str = "fail") -> None:
    """Land a punchline after a demo. `kind` is 'fail' (red) or 'win' (green)."""
    if kind == "fail":
        emoji, color = "❌", "red"
    elif kind == "win":
        emoji, color = "✅", "green"
    else:
        emoji, color = "💡", "yellow"
    console.print()
    console.print(
        Panel(
            Text(text, style=f"bold {color}"),
            border_style=color,
            title=f" {emoji} ",
            title_align="left",
            padding=(0, 2),
        )
    )
    console.print()


def step(num: int, title: str) -> None:
    """Mark a numbered step in a script."""
    console.print(f"\n[bold cyan]→ Step {num}.[/bold cyan] [bold]{title}[/bold]")


@contextmanager
def live_status(message: str) -> Iterator[None]:
    """Show a spinner while a slow operation runs."""
    with console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
        yield


def pause_for_effect(seconds: float = 0.6) -> None:
    """Brief silence between sections so the audience can read the previous one."""
    time.sleep(seconds)


# ── Health checks ─────────────────────────────────────────────────────────────


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
            RIME_ENDPOINT,
            headers={
                "Authorization": f"Bearer {key}",
                "Accept": "audio/mp3",
                "Content-Type": "application/json",
            },
            json={
                "speaker": "abbie",
                "text": ".",
                "modelId": "mistv2",
            },
            timeout=15,
        )
        if r.status_code == 200 and r.content and "json" not in r.headers.get("Content-Type", ""):
            console.print("[green]✓ Rime key valid[/green]")
            return True
        console.print(f"[red]✗ Rime returned HTTP {r.status_code}[/red]")
        return False
    except requests.RequestException as e:
        console.print(f"[red]✗ Rime check failed: {e}[/red]")
        return False


def check_nebius() -> bool:
    """Verify NEBIUS_API_KEY with a 1-token chat completion (cheapest valid call)."""
    key = get_key("NEBIUS_API_KEY", required=False)
    if not key:
        console.print("[red]✗ NEBIUS_API_KEY not set[/red]")
        return False
    try:
        client = openai.OpenAI(api_key=key, base_url=NEBIUS_BASE_URL, timeout=30.0, max_retries=1)
        client.chat.completions.create(
            model=DEFAULT_NEBIUS_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
        )
        console.print(
            f"[green]✓ Nebius key valid[/green] [dim](model: {DEFAULT_NEBIUS_MODEL})[/dim]"
        )
        return True
    except Exception as e:
        console.print(f"[red]✗ Nebius check failed: {e}[/red]")
        return False


def check_all() -> None:
    """Run all three checks and print a summary table.

    Exits with code 1 if any check fails.
    """
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Provider", style="white")
    table.add_column("Status", justify="center")

    results = {
        "Deepgram": check_deepgram(),
        "Rime": check_rime(),
        "Nebius": check_nebius(),
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
