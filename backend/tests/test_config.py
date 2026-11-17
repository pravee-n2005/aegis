"""Tests for Settings' OpenRouter configuration validation.

Uses Settings(...) directly (not the module-level `settings` singleton,
already constructed at import time from the real environment) so each test
controls its own inputs without needing to patch env vars process-wide.
"""
import os

import pytest

os.environ.setdefault("OPENROUTER_API_KEY", "test-key")

from pydantic import ValidationError  # noqa: E402

from app.config import Settings  # noqa: E402


def _base_kwargs(**overrides):
    kwargs = {"openrouter_api_key": "sk-or-real-key"}
    kwargs.update(overrides)
    return kwargs


def test_missing_openrouter_api_key_raises_clear_configuration_error():
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, openrouter_api_key="")
    message = str(exc_info.value)
    assert "OPENROUTER_API_KEY" in message
    assert "placeholder" in message or "missing" in message.lower()


def test_placeholder_openrouter_api_key_raises_clear_configuration_error():
    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None, openrouter_api_key="replace_me")
    message = str(exc_info.value)
    assert "OPENROUTER_API_KEY" in message
    assert "replace_me" in message
    # The error names the problem; it never echoes a real secret (there is
    # none here - "replace_me" is the placeholder value itself, not a key).


def test_real_openrouter_api_key_constructs_settings_without_error():
    settings = Settings(**_base_kwargs())
    assert settings.openrouter_api_key == "sk-or-real-key"


def test_default_openrouter_model_is_set():
    settings = Settings(**_base_kwargs())
    assert settings.openrouter_model
    assert settings.openrouter_model.endswith(":free")


def test_openrouter_model_overridable():
    settings = Settings(**_base_kwargs(openrouter_model="some-org/some-model"))
    assert settings.openrouter_model == "some-org/some-model"
