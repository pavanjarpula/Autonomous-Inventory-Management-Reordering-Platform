"""
Chooses which chat model backs the system, so the same code runs against a
local Ollama during development and a hosted API once deployed.

Everything downstream (the reasoning node, the chatbot, the cache wrapper)
only ever calls `.with_structured_output(schema).invoke(prompt)`. Every
provider below implements that identically, so switching is a config change,
not a code change.

Selection order: explicit argument, then SELLERSENSE_LLM_PROVIDER, then
whichever provider is actually usable on this machine (package installed and
key present), preferring a hosted API since local Ollama is unreachable from
a deployed container.

Provider packages are imported lazily and are all optional -- an uninstalled
provider raises a message telling you exactly what to install, rather than
breaking import for everyone else.

LangSmith integration: all functions are traceable via @traceable decorators.
Set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable tracing.
"""

import os
from dataclasses import dataclass

try:
    from langsmith import traceable
except ImportError:
    # Fallback: no-op decorator if langsmith is not installed
    def traceable(name=None, run_type="chain"):
        def decorator(func):
            return func
        return decorator


def _get_secret(key: str) -> str | None:
    """Check os.environ first, then Streamlit secrets (for cloud deployment)."""
    val = os.environ.get(key)
    if val:
        return val
    try:
        import streamlit as st
        return st.secrets.get(key)
    except Exception:
        return None

# model choices are per-provider defaults, overridable via SELLERSENSE_LLM_MODEL
PROVIDERS = {
    "groq":     dict(env_key="GROQ_API_KEY",   package="langchain_groq",         default_model="llama-3.3-70b-versatile"),
    "google":   dict(env_key="GOOGLE_API_KEY", package="langchain_google_genai", default_model="gemini-2.0-flash"),
    "openai":   dict(env_key="OPENAI_API_KEY", package="langchain_openai",       default_model="gpt-4o-mini"),
    "ollama":   dict(env_key=None,             package="langchain_ollama",       default_model="qwen3:4b"),
}

# hosted APIs first: a deployed container can reach them, and they're far faster
# than CPU inference. Ollama last, as the local-development fallback.
_PREFERENCE = ["groq", "google", "openai", "ollama"]


@dataclass
class ProviderStatus:
    name: str
    package_installed: bool
    key_present: bool

    @property
    def usable(self) -> bool:
        return self.package_installed and self.key_present


def _package_installed(package: str) -> bool:
    import importlib.util
    return importlib.util.find_spec(package) is not None


@traceable(name="provider_status", run_type="chain")
def provider_status(name: str) -> ProviderStatus:
    spec = PROVIDERS[name]
    return ProviderStatus(
        name=name,
        package_installed=_package_installed(spec["package"]),
        # Ollama needs no key; whether it's actually running is only knowable by
        # calling it, which callers already handle by catching the failure
        key_present=True if spec["env_key"] is None else bool(_get_secret(spec["env_key"])),
    )


@traceable(name="available_providers", run_type="chain")
def available_providers() -> list[str]:
    return [name for name in _PREFERENCE if provider_status(name).usable]


@traceable(name="resolve_provider", run_type="chain")
def resolve_provider(explicit: str | None = None) -> str:
    if explicit:
        if explicit not in PROVIDERS:
            raise ValueError(f"unknown provider {explicit!r} -- choose from {sorted(PROVIDERS)}")
        return explicit

    from_env = os.environ.get("SELLERSENSE_LLM_PROVIDER")
    if from_env:
        if from_env not in PROVIDERS:
            raise ValueError(f"SELLERSENSE_LLM_PROVIDER={from_env!r} is not one of {sorted(PROVIDERS)}")
        return from_env

    usable = available_providers()
    if not usable:
        raise RuntimeError(
            "No LLM provider is usable. Either set a hosted API key "
            "(GROQ_API_KEY / GOOGLE_API_KEY / OPENAI_API_KEY) and install its package, "
            "or run Ollama locally with `ollama serve`."
        )
    return usable[0]


@traceable(name="make_llm", run_type="llm")
def make_llm(provider: str | None = None, model: str | None = None, temperature: float = 0):
    """Returns a LangChain chat model. Same interface whichever provider backs it."""
    name = resolve_provider(provider)
    spec = PROVIDERS[name]
    model = model or os.environ.get("SELLERSENSE_LLM_MODEL") or spec["default_model"]

    status = provider_status(name)
    if not status.package_installed:
        raise RuntimeError(
            f"provider {name!r} needs its package: pip install {spec['package'].replace('_', '-')}"
        )
    if not status.key_present:
        raise RuntimeError(f"provider {name!r} needs {spec['env_key']} set in the environment")

    if name == "groq":
        from langchain_groq import ChatGroq
        api_key = _get_secret("GROQ_API_KEY")
        return ChatGroq(model=model, temperature=temperature, groq_api_key=api_key)
    if name == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = _get_secret("GOOGLE_API_KEY")
        return ChatGoogleGenerativeAI(model=model, temperature=temperature, google_api_key=api_key)
    if name == "openai":
        from langchain_openai import ChatOpenAI
        api_key = _get_secret("OPENAI_API_KEY")
        return ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
    if name == "ollama":
        from langchain_ollama import ChatOllama
        # base_url is settable for a remote Ollama (a tunnel, or a GPU box);
        # the default localhost is only reachable when running on the same machine
        base_url = os.environ.get("OLLAMA_BASE_URL")
        kwargs = dict(model=model, temperature=temperature)
        if base_url:
            kwargs["base_url"] = base_url
        return ChatOllama(**kwargs)

    raise ValueError(f"unhandled provider {name!r}")
