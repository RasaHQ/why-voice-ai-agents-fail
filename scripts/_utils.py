"""Shared utilities for the four failure-mode scripts.

This module holds the things every script needs so the failure-mode files
themselves stay focused on what they're demonstrating, not on plumbing.

Specifically:

1. Loads API credentials from `.env`, the environment, or Colab secrets.
2. Provides a single, friendly Rich console for consistent script output.
3. Provides a shared `make_llm_client()` factory that returns a Nebius-backed
   OpenAI client — with sensible timeouts and retries.
4. Provides `synthesize()` / `synthesize_with_pause()` / `play()` helpers so
   every script speaks audio the same way and plays it through the speakers.
5. Provides projector-friendly narrative helpers (`header`, `verdict`,
   `narrate`, `live_status`) so the scripts feel like a presentation rather
   than a wall of tables.
6. Provides cheap "is this credential actually valid?" health checks
   that the Makefile and `verify_setup.py` call before running the scripts.

Why MP3 for most scripts but WAV for `synthesize_with_pause()`? MP3 plays
everywhere out of the box and Deepgram ingests it natively. But splicing
silence into MP3 reliably requires ffmpeg, which we don't want as a hard
dep. WAV manipulation, on the other hand, is in Python's stdlib (`wave`).
So we use WAV when we need to manipulate audio, MP3 otherwise.
"""

from __future__ import annotations

import io
import os
import platform
import shutil
import subprocess
import sys
import time
import wave
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
DEFAULT_NEBIUS_MODEL: Final[str] = os.environ.get(
    "NEBIUS_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"
)
PLAY_AUDIO: Final[bool] = os.environ.get("PLAY_AUDIO", "true").lower() != "false"

# ── Console ───────────────────────────────────────────────────────────────────
console: Final[Console] = Console()


# ── Credential loading ────────────────────────────────────────────────────────
def get_key(name: str, *, required: bool = True) -> str | None:
    """Load an API key from Colab secrets, .env, or the environment."""
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
    """Return an OpenAI SDK client wired up to Nebius Token Factory."""
    api_key = get_key("NEBIUS_API_KEY")
    return openai.OpenAI(
        api_key=api_key,
        base_url=NEBIUS_BASE_URL,
        timeout=timeout,
        max_retries=max_retries,
    )


# ── TTS via Rime ──────────────────────────────────────────────────────────────
RIME_ENDPOINT: Final[str] = "https://users.rime.ai/v1/rime-tts"


def _rime_post(
    text: str,
    *,
    audio_format: str,
    speaker: str,
    model_id: str,
    speed_alpha: float,
    sampling_rate: int = 22050,
) -> bytes:
    """Low-level Rime call. Returns raw audio bytes."""
    rime_key = get_key("RIME_API_KEY")
    accept = {"mp3": "audio/mp3", "wav": "audio/wav"}.get(audio_format, "audio/mp3")

    payload = {
        "speaker": speaker,
        "text": text,
        "modelId": model_id,
        "speedAlpha": speed_alpha,
    }
    if audio_format == "wav":
        payload["samplingRate"] = sampling_rate

    response = requests.post(
        RIME_ENDPOINT,
        headers={
            "Authorization": f"Bearer {rime_key}",
            "Accept": accept,
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "json" in content_type or len(response.content) < 256:
        raise RuntimeError(
            f"Rime returned non-audio response (Content-Type={content_type!r}, "
            f"size={len(response.content)} bytes): {response.text[:200]}"
        )
    return response.content


def synthesize(
    text: str,
    output_path: Path,
    *,
    speaker: str = "abbie",
    model_id: str = "mistv2",
    speed_alpha: float = 1.0,
    audio_format: str = "mp3",
) -> Path:
    """Generate audio from text using Rime TTS. Returns the output path."""
    audio_bytes = _rime_post(
        text,
        audio_format=audio_format,
        speaker=speaker,
        model_id=model_id,
        speed_alpha=speed_alpha,
    )
    output_path.write_bytes(audio_bytes)
    return output_path


def synthesize_with_pause(
    text_part1: str,
    text_part2: str,
    output_path: Path,
    *,
    silence_ms: int = 800,
    speaker: str = "abbie",
    model_id: str = "mistv2",
    speed_alpha: float = 1.0,
) -> Path:
    """Synthesize two text halves and splice GUARANTEED silence between them.

    This is what we use for Failure 01 — Rime's natural pauses on "..." aren't
    reliable enough to force Deepgram to split the audio into two utterances,
    so we generate the two halves separately and inject a deterministic
    silence gap.

    Uses WAV throughout (Rime returns WAV, stdlib `wave` mixes them) — no
    ffmpeg required. The output WAV is rebuilt by Python's wave module, so
    its header is canonical and Deepgram parses it without complaint.
    """
    sampling_rate = 22050  # Rime's WAV default

    part1 = _rime_post(
        text_part1,
        audio_format="wav",
        speaker=speaker,
        model_id=model_id,
        speed_alpha=speed_alpha,
        sampling_rate=sampling_rate,
    )
    part2 = _rime_post(
        text_part2,
        audio_format="wav",
        speaker=speaker,
        model_id=model_id,
        speed_alpha=speed_alpha,
        sampling_rate=sampling_rate,
    )

    # Read each WAV via stdlib `wave`. This also normalises the headers —
    # Rime's RIFF chunk is sometimes non-canonical, but `wave` reads the
    # essentials and our re-write uses a clean header.
    with wave.open(io.BytesIO(part1), "rb") as w1:
        params = w1.getparams()
        frames1 = w1.readframes(w1.getnframes())
    with wave.open(io.BytesIO(part2), "rb") as w2:
        frames2 = w2.readframes(w2.getnframes())

    # Generate silence at the same params as part 1.
    n_silence_samples = int(params.framerate * silence_ms / 1000)
    silence = b"\x00" * (n_silence_samples * params.sampwidth * params.nchannels)

    with wave.open(str(output_path), "wb") as out:
        out.setparams(params)
        out.writeframes(frames1)
        out.writeframes(silence)
        out.writeframes(frames2)

    return output_path


# ── Audio playback ────────────────────────────────────────────────────────────
_MAC_PLAYER = "afplay"
_LINUX_PLAYERS = ["mpg123", "ffplay", "play", "paplay", "aplay"]


def _find_linux_player() -> list[str] | None:
    """Return argv for a Linux audio player that's installed."""
    for cmd in _LINUX_PLAYERS:
        if shutil.which(cmd):
            if cmd == "ffplay":
                return ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]
            if cmd == "play":
                return ["play", "-q"]
            return [cmd]
    return None


def play(path: Path, *, label: str | None = None, blocking: bool = True) -> None:
    """Play an audio file through the host's default speakers.

    `blocking=True` (default) waits for the playback to finish before returning.
    Honors the PLAY_AUDIO env var — set `PLAY_AUDIO=false` to skip playback.
    """
    if label:
        console.print(f"  [bold magenta]🔊 {label}[/bold magenta]")

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
        argv = ["cmd.exe", "/c", "start", "/wait", "", str(path)]

    if argv is None:
        console.print(
            "  [yellow]⚠ No audio player found.[/yellow] "
            "[dim]Install: brew install mpg123 (mac), apt install mpg123 (linux).[/dim]"
        )
        return

    try:
        if blocking:
            subprocess.run(
                argv,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    except (FileNotFoundError, OSError) as e:
        console.print(f"  [yellow]⚠ Audio playback failed: {e}[/yellow]")


# ── Narrative helpers ─────────────────────────────────────────────────────────
# Designed for a projector. Each helper is meant to dominate the visible
# screen at the moment it's called — the audience should be reading ONE
# thing at a time, not scanning a wall of text.


def header(title: str, subtitle: str = "") -> None:
    """Open a script with a banner panel."""
    body = Text(title, style="bold white")
    if subtitle:
        body.append("\n\n" + subtitle, style="dim italic")
    console.print()
    console.print(Panel(body, border_style="cyan", padding=(1, 4)))
    console.print()


def section(title: str) -> None:
    """Mark a major section."""
    console.print()
    console.rule(f"[bold cyan] {title} [/bold cyan]", style="cyan")
    console.print()


def narrate(text: str, *, style: str = "white") -> None:
    """One short sentence between demos. Keep it under ~140 characters."""
    console.print(f"  [{style}]{text}[/{style}]")


def step(num: int, title: str) -> None:
    """Mark a numbered step."""
    console.print(f"\n[bold cyan]Step {num}.[/bold cyan] [bold]{title}[/bold]")


def watch_this(text: str) -> None:
    """A 'watch this' announcement before the demo runs."""
    console.print()
    console.print(
        Panel(
            Text(text, style="bold yellow", justify="left"),
            border_style="yellow",
            title="[bold yellow] 👀  WATCH THIS [/bold yellow]",
            title_align="left",
            padding=(0, 2),
        )
    )
    console.print()


def verdict(
    headline: str,
    detail: str = "",
    *,
    kind: str = "fail",
) -> None:
    """Big projector-friendly result panel.

    `kind`: 'fail' (red ❌), 'win' (green ✅), 'info' (yellow 💡).

    The headline goes in big bold. The detail is one or two short follow-up
    lines. Designed to be readable from row 30 of the room.
    """
    if kind == "fail":
        emoji, color, title = "❌", "red", "FAILURE — this is the point of the demo"
    elif kind == "win":
        emoji, color, title = "✅", "green", "SUCCESS — the fix worked"
    else:
        emoji, color, title = "💡", "yellow", "INCONCLUSIVE — read the detail"

    body = Text(headline, style=f"bold {color}", justify="left")
    if detail:
        body.append("\n\n" + detail, style="white")

    console.print()
    console.print(
        Panel(
            body,
            border_style=color,
            title=f"[bold {color}] {emoji}  {title} [/bold {color}]",
            title_align="left",
            padding=(1, 3),
        )
    )
    console.print()


def big_compare(left_label: str, left: str, right_label: str, right: str) -> None:
    """Two-column compare, projector-readable. Used for naive vs two-tier, etc."""
    table = Table.grid(padding=(0, 4))
    table.add_column(justify="center")
    table.add_column(justify="center")
    table.add_row(
        Panel(
            Text(left, style="bold red", justify="center"),
            title=f"[bold red] {left_label} [/bold red]",
            border_style="red",
            padding=(1, 2),
        ),
        Panel(
            Text(right, style="bold green", justify="center"),
            title=f"[bold green] {right_label} [/bold green]",
            border_style="green",
            padding=(1, 2),
        ),
    )
    console.print()
    console.print(table)
    console.print()


@contextmanager
def live_status(message: str) -> Iterator[None]:
    """Show a spinner while a slow operation runs."""
    with console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
        yield


def pause_for_effect(seconds: float = 0.6) -> None:
    """Brief silence between sections."""
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
    """Verify RIME_API_KEY with a 1-character TTS request."""
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
            json={"speaker": "abbie", "text": ".", "modelId": "mistv2"},
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
    """Verify NEBIUS_API_KEY with a 1-token chat completion."""
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
    """Run all three checks and print a summary table. Exits 1 on failure."""
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
