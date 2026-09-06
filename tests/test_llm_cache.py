import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_cache import CachedLLM, _cache_key


class Answer(BaseModel):
    text: str


class _RealStructured:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def invoke(self, prompt):
        self.calls += 1
        return self.response


class _RealLLM:
    def __init__(self, response):
        self.structured = _RealStructured(response)

    def with_structured_output(self, schema):
        return self.structured


@pytest.fixture
def cache_path(tmp_path):
    return tmp_path / "cache.json"


def test_cache_key_is_stable_for_the_same_inputs():
    assert _cache_key("Answer", "hello") == _cache_key("Answer", "hello")


def test_cache_key_differs_for_different_prompts():
    assert _cache_key("Answer", "hello") != _cache_key("Answer", "goodbye")


def test_replay_mode_returns_the_cached_value_without_touching_the_real_llm(cache_path):
    key = _cache_key("Answer", "why?")
    cache_path.write_text(json.dumps({key: {"text": "cached answer"}}))
    real = _RealLLM(Answer(text="should not be used"))

    llm = CachedLLM(cache_path, real_llm=real, record=False)
    result = llm.with_structured_output(Answer).invoke("why?")

    assert result.text == "cached answer"
    assert real.structured.calls == 0  # never touched the real model


def test_cache_miss_falls_through_to_the_real_llm_when_one_is_given(cache_path):
    real = _RealLLM(Answer(text="live answer"))
    llm = CachedLLM(cache_path, real_llm=real, record=False)
    result = llm.with_structured_output(Answer).invoke("a question not in the cache")
    assert result.text == "live answer"
    assert real.structured.calls == 1


def test_cache_miss_with_no_real_llm_raises_not_silently_returns_nothing(cache_path):
    llm = CachedLLM(cache_path, real_llm=None, record=False)
    with pytest.raises(RuntimeError):
        llm.with_structured_output(Answer).invoke("anything")


def test_record_mode_calls_the_real_llm_and_saves_the_response(cache_path):
    real = _RealLLM(Answer(text="freshly recorded"))
    llm = CachedLLM(cache_path, real_llm=real, record=True)
    result = llm.with_structured_output(Answer).invoke("a new question")

    assert result.text == "freshly recorded"
    assert real.structured.calls == 1
    assert len(llm) == 1

    llm.save()
    reloaded = json.loads(cache_path.read_text())
    assert list(reloaded.values())[0]["text"] == "freshly recorded"


def test_a_second_recording_session_can_replay_what_a_first_one_saved(cache_path):
    real1 = _RealLLM(Answer(text="original"))
    first = CachedLLM(cache_path, real_llm=real1, record=True)
    first.with_structured_output(Answer).invoke("the demo question")
    first.save()

    real2 = _RealLLM(Answer(text="should not be called"))
    second = CachedLLM(cache_path, real_llm=real2, record=False)
    result = second.with_structured_output(Answer).invoke("the demo question")

    assert result.text == "original"
    assert real2.structured.calls == 0
