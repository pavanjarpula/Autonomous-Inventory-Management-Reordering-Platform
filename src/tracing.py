"""
Unified tracing utility for LangSmith.

Builds nested traces with parent-child relationships using the explicit
Client API (not the @traceable decorator which produces flat traces).

Usage:
    from tracing import trace_run

    with trace_run("chat_question", inputs={"q": "..."}, metadata={"provider": "groq"}) as run:
        # child spans automatically nest under this parent
        with trace_run("classify_intent", run_type="llm", metadata={"item": "Rice"}) as child:
            ...
            child.set_outputs(intent="question")
        with trace_run("answer_question", run_type="llm") as child:
            ...
            child.set_outputs(answer="...")
        run.set_outputs(answer="...")
"""

from contextlib import contextmanager
from typing import Any, Optional
import os

try:
    from langsmith import Client as _LangSmithClient
except ImportError:
    _LangSmithClient = None

_client: Optional[Any] = None
_unavailable = False  # set True once we know LangSmith can't connect


def _get_client():
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is None and _LangSmithClient is not None:
        api_key = os.environ.get("LANGCHAIN_API_KEY")
        if not api_key:
            _unavailable = True
            return None
        try:
            _client = _LangSmithClient(auto_batch_tracing=False)
        except Exception:
            _unavailable = True
            return None
    return _client


def _project_name() -> str:
    return os.environ.get("LANGCHAIN_PROJECT", "sellersense")


@contextmanager
def trace_run(
    name: str,
    run_type: str = "chain",
    inputs: Optional[dict] = None,
    metadata: Optional[dict] = None,
    tags: Optional[list[str]] = None,
    parent_run_id: Optional[Any] = None,
):
    """
    Context manager that creates a LangSmith run and yields it so callers
    can attach outputs/extra metadata via _extra_outputs / _extra_meta.

    On exception the run is marked as error automatically.
    """
    client = _get_client()
    if client is None:
        # Tracing unavailable — yield a no-op object
        yield _NoopRun()
        return

    try:
        run = client.create_run(
            name=name,
            run_type=run_type,
            inputs=inputs or {},
            project_name=_project_name(),
            metadata=metadata or {},
            tags=tags or [],
            parent_run_id=parent_run_id,
        )
    except Exception:
        yield _NoopRun()
        return

    try:
        yield run
        if run and run.id:
            extra_outputs = getattr(run, "_extra_outputs", {})
            extra_meta = getattr(run, "_extra_meta", {})
            update_kwargs: dict[str, Any] = {}
            if extra_outputs:
                update_kwargs["outputs"] = extra_outputs
            if extra_meta:
                update_kwargs["extra_metadata"] = extra_meta
            if update_kwargs:
                client.update_run(run.id, **update_kwargs)
    except Exception as exc:
        if run and run.id:
            try:
                client.update_run(run.id, error=str(exc))
            except Exception:
                pass
        raise


class _NoopRun:
    """Dummy run when LangSmith is unavailable. Supports .id and attribute access."""
    id = None
    def __getattr__(self, name):
        return None
    def set_outputs(self, **kw): pass
    def update(self, **kw): pass
