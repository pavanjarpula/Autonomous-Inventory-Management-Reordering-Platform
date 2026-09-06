"""
Demo-safety caching. Wraps a real LLM so a live model call is never
a single point of failure during the actual walkthrough: record real responses
once during rehearsal, replay them by exact prompt match during the live demo.
A cache miss falls through to a live call if one's available, or raises --
which every existing caller (graph.py's reasoning node, chatbot.py) already
catches and degrades from, the same retry/fallback path a live-model failure
would trigger anyway. No new failure mode, just one more path into the one
that already exists.

Same with_structured_output(schema).invoke(prompt) -> schema interface as
ChatOllama, so this is a drop-in swap everywhere an LLM is passed in.
"""

import hashlib
import json
from pathlib import Path

from pydantic import BaseModel


def _cache_key(schema_name: str, prompt: str) -> str:
    return hashlib.sha256(f"{schema_name}::{prompt}".encode()).hexdigest()[:16]


class _CachedStructured:
    def __init__(self, schema: type[BaseModel], entries: dict, real_structured, record: bool):
        self.schema = schema
        self.entries = entries
        self.real_structured = real_structured
        self.record = record

    def invoke(self, prompt: str):
        key = _cache_key(self.schema.__name__, prompt)
        if key in self.entries:
            return self.schema(**self.entries[key])
        if self.real_structured is None:
            raise RuntimeError(f"cache miss for {self.schema.__name__} and no live LLM to fall back to")
        result = self.real_structured.invoke(prompt)
        if self.record:
            self.entries[key] = result.model_dump()
        return result


class CachedLLM:
    """
    use_cache=True, record=False (the demo default): replay from the saved
    cache file, fall through to `real_llm` (if given) on a miss.
    record=True: call `real_llm` for real and save every response -- this is
    what the rehearsal recording script uses to build the cache file.
    """
    def __init__(self, cache_path: Path, real_llm=None, record: bool = False):
        self.cache_path = Path(cache_path)
        self.real_llm = real_llm
        self.record = record
        self.entries: dict = json.loads(self.cache_path.read_text()) if self.cache_path.exists() else {}

    def with_structured_output(self, schema: type[BaseModel]):
        real_structured = self.real_llm.with_structured_output(schema) if self.real_llm is not None else None
        return _CachedStructured(schema, self.entries, real_structured, self.record)

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.entries, indent=2))

    def __len__(self):
        return len(self.entries)
