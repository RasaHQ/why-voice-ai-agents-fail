"""Smoke tests for the shared utilities.

These tests don't make network calls. They verify that the scripts
package is importable, the credential resolver behaves correctly,
and the audio output directory can be created.
"""

from __future__ import annotations

import pytest

from scripts import _utils


def test_module_imports_cleanly():
    """The _utils module imports without raising."""
    assert _utils.console is not None
    assert _utils.AUDIO_OUTPUT_DIR.is_absolute()
    assert _utils.REPO_ROOT.is_absolute()


def test_get_key_returns_env_value(monkeypatch):
    """get_key reads from os.environ when no Colab is present."""
    monkeypatch.setenv("FAKE_TEST_KEY", "value-123")
    assert _utils.get_key("FAKE_TEST_KEY") == "value-123"


def test_get_key_required_missing_raises(monkeypatch):
    """get_key raises when required=True and key is not set."""
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    with pytest.raises(RuntimeError, match="DEFINITELY_NOT_SET is not set"):
        _utils.get_key("DEFINITELY_NOT_SET", required=True)


def test_get_key_optional_missing_returns_none(monkeypatch):
    """get_key returns None when required=False and key is not set."""
    monkeypatch.delenv("DEFINITELY_NOT_SET", raising=False)
    assert _utils.get_key("DEFINITELY_NOT_SET", required=False) is None


def test_ensure_audio_dir_creates_directory(tmp_path, monkeypatch):
    """ensure_audio_dir creates the directory and returns its path."""
    fake_dir = tmp_path / "audio"
    monkeypatch.setattr(_utils, "AUDIO_OUTPUT_DIR", fake_dir)
    result = _utils.ensure_audio_dir()
    assert result == fake_dir
    assert fake_dir.is_dir()


def test_check_deepgram_returns_false_without_key(monkeypatch):
    """check_deepgram returns False when DEEPGRAM_API_KEY is not set."""
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    assert _utils.check_deepgram() is False


def test_check_rime_returns_false_without_key(monkeypatch):
    """check_rime returns False when RIME_API_KEY is not set."""
    monkeypatch.delenv("RIME_API_KEY", raising=False)
    assert _utils.check_rime() is False


def test_check_openai_returns_false_without_key(monkeypatch):
    """check_openai returns False when OPENAI_API_KEY is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert _utils.check_openai() is False
