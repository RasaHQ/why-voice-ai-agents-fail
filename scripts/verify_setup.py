#!/usr/bin/env python3
"""Pre-flight diagnostics for the four failure-mode scripts.

Checks everything the scripts need before they run, so problems surface
once at the start of the session instead of three minutes into a script.

Verified:
  - Python version (3.11+)
  - .env loads cleanly (or warns if missing)
  - Required environment variables (DEEPGRAM_API_KEY, RIME_API_KEY, NEBIUS_API_KEY)
  - Python dependencies (deepgram-sdk, openai, requests, rich, python-dotenv)
  - Package layout (scripts/ is importable)
  - External service connectivity (Deepgram, Rime, Nebius — real round-trips)

Usage::

    make verify
    # or directly:
    python scripts/verify_setup.py

Exit codes:
  0  all checks passed (or only warnings)
  1  one or more required checks failed
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

# Make sure we can import `scripts._utils` no matter where we're invoked from.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._utils import (  # noqa: E402  (sys.path edit must come first)
    DEFAULT_NEBIUS_MODEL,
    NEBIUS_BASE_URL,
    check_deepgram,
    check_nebius,
    check_rime,
    console,
)

# ── Output helpers ────────────────────────────────────────────────────────────
# Rich gives us colour and structure. We define short aliases so the check
# functions read cleanly.


def section(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]", style="cyan")


def ok(msg: str) -> None:
    console.print(f"  [green]✓[/green]  {msg}")


def warn(msg: str) -> None:
    console.print(f"  [yellow]⚠[/yellow]  {msg}")


def fail(msg: str) -> None:
    console.print(f"  [red]✗[/red]  {msg}")


def hint(msg: str) -> None:
    console.print(f"     [dim]→ {msg}[/dim]")


def info(msg: str) -> None:
    console.print(f"  [blue]ℹ[/blue]  {msg}")  # noqa: RUF001 (deliberate info glyph)


# ── Individual checks ─────────────────────────────────────────────────────────


def check_python_version() -> bool:
    """Python 3.11+ is required (matches pyproject.toml)."""
    v = sys.version_info
    if v.major == 3 and v.minor >= 11:
        ok(f"Python {v.major}.{v.minor}.{v.micro}")
        return True
    fail(f"Python {v.major}.{v.minor} detected — requires 3.11 or newer")
    hint("With pyenv: pyenv install 3.12 && pyenv local 3.12")
    hint("Or with uv:  uv venv --python 3.12 .venv")
    return False


def check_env_file() -> bool:
    """Inform the user whether .env was found."""
    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        ok(f".env file found at {env_path.relative_to(REPO_ROOT)}")
        return True
    warn(".env file not found at repo root")
    hint("Copy .env.example to .env and fill in your keys")
    return False


def _mask(value: str) -> str:
    """Return value masked to first 4 + last 4 chars for safe display."""
    if len(value) <= 8:
        return "***"
    return f"{value[:4]}...{value[-4:]}"


def check_env_var(name: str, label: str) -> bool:
    """Check an env var is set and not still a placeholder."""
    value = os.environ.get(name, "").strip()
    if not value:
        fail(f"{label}  ({name}) not set")
        hint(f"Add {name}=your-key to your .env file")
        return False
    if value.lower().startswith("your-") or "here" in value.lower():
        fail(f"{label}  ({name}) still has the placeholder value")
        hint("Replace the placeholder in .env with your real key")
        return False
    ok(f"{label}  [dim]({name}={_mask(value)})[/dim]")
    return True


def check_module(module: str, label: str) -> bool:
    """Verify a Python module is importable."""
    if importlib.util.find_spec(module) is not None:
        ok(label)
        return True
    fail(f"{label}  ({module}) not installed")
    hint("Run: make install")
    return False


def check_scripts_package() -> bool:
    """Verify the local scripts package layout is intact."""
    needed = [
        ("scripts", "scripts  (package)"),
        ("scripts._utils", "scripts._utils  (shared helpers)"),
    ]
    all_ok = True
    for module, label in needed:
        if not check_module(module, label):
            all_ok = False
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────


def run_checks() -> int:
    """Run all checks and return an exit code (0 = success, 1 = failure)."""
    console.print()
    console.rule(
        "[bold magenta]🎙  Why (Voice) AI Agents Fail — pre-flight[/bold magenta]",
        style="magenta",
    )
    info("All scripts run in-process. No servers required.")

    errors = 0
    warnings = 0

    # ── Python ────────────────────────────────────────────────────────────────
    section("Python environment")
    if not check_python_version():
        errors += 1

    if not check_env_file():
        warnings += 1  # warning, not fatal — env vars might come from elsewhere

    # ── API keys ──────────────────────────────────────────────────────────────
    section("API keys (.env)")
    if not check_env_var("DEEPGRAM_API_KEY", "Deepgram API key   (ASR)"):
        errors += 1
    if not check_env_var("RIME_API_KEY", "Rime API key       (TTS)"):
        errors += 1
    if not check_env_var("NEBIUS_API_KEY", "Nebius API key     (LLM)"):
        errors += 1

    info(f"Default Nebius model: [bold]{DEFAULT_NEBIUS_MODEL}[/bold]")
    info(f"Default Nebius endpoint: [dim]{NEBIUS_BASE_URL}[/dim]")

    # ── Python deps ───────────────────────────────────────────────────────────
    section("Python dependencies")
    deps = [
        ("deepgram", "deepgram-sdk"),
        ("openai", "openai  (SDK used for Nebius too)"),
        ("requests", "requests"),
        ("rich", "rich"),
        ("dotenv", "python-dotenv"),
    ]
    for module, label in deps:
        if not check_module(module, label):
            errors += 1

    # ── Package layout ────────────────────────────────────────────────────────
    section("Package layout")
    if not check_scripts_package():
        errors += 1

    # ── External services ────────────────────────────────────────────────────
    # Only run the network round-trips if the keys passed presence checks.
    section("External service connectivity")
    if errors == 0:
        if not check_deepgram():
            errors += 1
        if not check_rime():
            errors += 1
        if not check_nebius():
            errors += 1
    else:
        info("Skipped — fix the errors above first, then re-run")

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    console.rule(style="cyan")

    if errors == 0 and warnings == 0:
        console.print("[bold green]✓  All checks passed — ready to run the scripts.[/bold green]")
        console.print()
        console.print("  [magenta]Run all four scripts:[/magenta]")
        console.print("    [green]make scripts[/green]")
        console.print()
        console.print("  [magenta]Run them individually:[/magenta]")
        console.print("    [green]make script-01[/green]   Turn-taking")
        console.print("    [green]make script-02[/green]   Backchannels vs interrupts")
        console.print("    [green]make script-03[/green]   Split state")
        console.print("    [green]make script-04[/green]   Premature goodbye")

    elif errors == 0:
        console.print(
            f"[bold yellow]⚠  Ready with {warnings} warning(s) — see above.[/bold yellow]"
        )
        console.print()
        console.print("  [magenta]Run all four scripts:[/magenta]")
        console.print("    [green]make scripts[/green]")

    else:
        console.print(
            f"[bold red]✗  {errors} error(s) — fix them before running the scripts.[/bold red]"
        )
        if warnings:
            console.print(f"[yellow]  Also {warnings} warning(s) noted above.[/yellow]")
        console.print()
        console.print("  [blue]Common fixes:[/blue]")
        console.print("    [green]make install[/green]            install / reinstall dependencies")
        console.print("    [green]cp .env.example .env[/green]   then fill in your API keys")

    console.print()
    return 0 if errors == 0 else 1


def main() -> None:
    sys.exit(run_checks())


if __name__ == "__main__":
    main()
