"""
Minimal LangSmith tracing helper.

Creates one clean nested trace per operation using the Client API.
No @traceable decorators — just explicit parent-child spans.

Usage:
    from tracing import trace_run

    with trace_run("chat_question", inputs={"q": "..."}) as run:
        with trace_run("respond", parent_run_id=run.id) as child:
            child._extra_outputs = {"answer": "..."}
"""

from contextlib import contextmanager
from typing import Any, Optional
import os

_client = None
_unavailable = False


def _get_client():
    global _client, _unavailable
    if _unavailable:
        return None
    if _client is None:
        try:
            from langsmith import Client
            api_key = os.environ.get("LANGCHAIN_API_KEY")
            if not api_key:
                _unavailable = True
                return None
            _client = Client(auto_batch_tracing=False)
        except Exception:
            _unavailable = True
            return None
    return _client


class _NoopRun:
    """Dummy run when LangSmith is unavailable."""
    id = None
    def __getattr__(self, name):
        return None
    def __setattr__(self, name, value):
        pass


@contextmanager
def trace_run(name, run_type="chain", inputs=None, metadata=None,
              tags=None, parent_run_id=None):
    """Create a LangSmith run. Yields a run object with ._extra_outputs."""
    client = _get_client()
    if client is None:
        yield _NoopRun()
        return

    try:
        run = client.create_run(
            name=name, run_type=run_type,
            inputs=inputs or {},
            project_name=os.environ.get("LANGCHAIN_PROJECT", "sellersense"),
            metadata=metadata or {}, tags=tags or [],
            parent_run_id=parent_run_id,
        )
    except Exception:
        yield _NoopRun()
        return

    if run is None:
        yield _NoopRun()
        return

    try:
        yield run
        extras = getattr(run, "_extra_outputs", {})
        if extras:
            client.update_run(run.id, outputs=extras)
    except Exception as exc:
        try:
            client.update_run(run.id, error=str(exc))
        except Exception:
            pass
        raise
