import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import llm_provider
from llm_provider import PROVIDERS, available_providers, make_llm, resolve_provider


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Every test starts from a known environment -- otherwise a real key on the
    developer's machine silently changes what these assert."""
    for spec in PROVIDERS.values():
        if spec["env_key"]:
            monkeypatch.delenv(spec["env_key"], raising=False)
    monkeypatch.delenv("SELLERSENSE_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("SELLERSENSE_LLM_MODEL", raising=False)
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)


def _all_packages_present(monkeypatch):
    monkeypatch.setattr(llm_provider, "_package_installed", lambda package: True)


# ---------------------------------------------------------------- resolution

def test_explicit_provider_wins(monkeypatch):
    monkeypatch.setenv("SELLERSENSE_LLM_PROVIDER", "google")
    assert resolve_provider("openai") == "openai"


def test_env_var_is_used_when_no_explicit_choice(monkeypatch):
    monkeypatch.setenv("SELLERSENSE_LLM_PROVIDER", "google")
    assert resolve_provider() == "google"


def test_unknown_provider_is_rejected_rather_than_silently_defaulting():
    with pytest.raises(ValueError):
        resolve_provider("not-a-provider")


def test_unknown_provider_in_env_is_rejected(monkeypatch):
    monkeypatch.setenv("SELLERSENSE_LLM_PROVIDER", "typo-provider")
    with pytest.raises(ValueError):
        resolve_provider()


def test_hosted_api_is_preferred_over_local_ollama(monkeypatch):
    # a deployed container can reach an API but not a localhost Ollama, so when
    # both are usable the hosted one must win by default
    _all_packages_present(monkeypatch)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    assert resolve_provider() == "groq"


def test_ollama_is_the_fallback_when_no_api_key_is_set(monkeypatch):
    _all_packages_present(monkeypatch)
    assert resolve_provider() == "ollama"  # needs no key


def test_available_providers_lists_only_usable_ones(monkeypatch):
    _all_packages_present(monkeypatch)
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    usable = available_providers()
    assert "google" in usable
    assert "ollama" in usable      # no key required
    assert "groq" not in usable    # package present but no key
    assert "openai" not in usable


def test_nothing_usable_raises_an_actionable_error(monkeypatch):
    monkeypatch.setattr(llm_provider, "_package_installed", lambda package: False)
    with pytest.raises(RuntimeError, match="No LLM provider is usable"):
        resolve_provider()


# ---------------------------------------------------------------- construction errors

def test_missing_package_names_the_pip_install(monkeypatch):
    monkeypatch.setattr(llm_provider, "_package_installed", lambda package: False)
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    with pytest.raises(RuntimeError, match="pip install langchain-groq"):
        make_llm("groq")


def test_missing_key_names_the_env_var(monkeypatch):
    _all_packages_present(monkeypatch)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        make_llm("groq")
