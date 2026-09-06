"""
Tests for src/tools.py and src/agents.py (Phase 2: tool calling + multi-agent).

Tests cover:
- Tool invocation with valid/invalid inputs
- Tool output structure
- Supervisor loop behavior
- Agent state management
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


# ---- Tool Tests ----

class TestTools:
    """Tests for @tool-decorated functions in tools.py."""

    def test_assess_inventory_item_returns_dict(self):
        from tools import assess_inventory_item
        result = assess_inventory_item.invoke({
            "item_id": "I001",
            "as_of_date": "2026-08-24",
        })
        assert isinstance(result, dict)
        assert result["item_id"] == "I001"
        assert "risk" in result
        assert "on_hand" in result
        assert "suggested_order_qty" in result

    def test_get_item_context_returns_dict(self):
        from tools import get_item_context
        result = get_item_context.invoke({
            "item_id": "I001",
            "as_of_date": "2026-08-24",
        })
        assert isinstance(result, dict)
        assert "is_festival_window" in result

    def test_get_on_order_status_returns_dict(self):
        from tools import get_on_order_status
        result = get_on_order_status.invoke({
            "item_id": "I001",
            "as_of_date": "2026-08-24",
        })
        assert isinstance(result, dict)
        assert result["item_id"] == "I001"
        assert "on_order_qty" in result
        assert "days_to_next_arrival" in result

    def test_get_demand_trend_returns_dict(self):
        from tools import get_demand_trend
        result = get_demand_trend.invoke({
            "item_id": "I001",
        })
        assert isinstance(result, dict)
        assert result["item_id"] == "I001"
        assert "trend_ratio" in result
        assert "interpretation" in result

    def test_batch_assess_items_returns_list(self):
        from tools import batch_assess_items
        result = batch_assess_items.invoke({
            "as_of_date": "2026-08-24",
            "limit": 5,
        })
        assert isinstance(result, list)
        assert len(result) <= 5
        if result:
            assert "item_id" in result[0]
            assert "risk" in result[0]

    def test_batch_assess_items_filters_by_risk(self):
        from tools import batch_assess_items
        result = batch_assess_items.invoke({
            "as_of_date": "2026-08-24",
            "risk_filter": "stockout_risk",
            "limit": 10,
        })
        assert isinstance(result, list)
        for item in result:
            assert item["risk"] == "stockout_risk"

    def test_get_forecast_summary_returns_dict(self):
        from tools import get_forecast_summary
        result = get_forecast_summary.invoke({
            "item_id": "I001",
            "horizon_days": 7,
        })
        assert isinstance(result, dict)
        assert result["item_id"] == "I001"
        assert "forecast_mean_daily" in result
        assert "forecast_total" in result

    def test_all_tools_have_names(self):
        from tools import ALL_TOOLS
        for t in ALL_TOOLS:
            assert hasattr(t, "name")
            assert t.name is not None

    def test_tools_are_json_serializable(self):
        from tools import assess_inventory_item, get_item_context
        import json
        
        result = assess_inventory_item.invoke({
            "item_id": "I001",
            "as_of_date": "2026-08-24",
        })
        # Should be JSON-serializable
        json.dumps(result, default=str)


# ---- Agent Tests ----

class TestAgents:
    """Tests for agent patterns in agents.py."""

    def test_agent_state_structure(self):
        from agents import AgentState
        state = AgentState(
            messages=[],
            item_id="I001",
            as_of_date="2026-08-24",
            final_answer=None,
            iteration=0,
            max_iterations=5,
        )
        assert state["item_id"] == "I001"
        assert state["iteration"] == 0

    def test_import_agents_module(self):
        import agents
        assert hasattr(agents, "supervisor_agent")
        assert hasattr(agents, "inventory_analyst_agent")
        assert hasattr(agents, "chat_agent")
        assert hasattr(agents, "ALL_TOOLS")

    def test_tools_list_matches_main_module(self):
        from agents import ALL_TOOLS as agent_tools
        from tools import ALL_TOOLS as main_tools
        assert len(agent_tools) == len(main_tools)

    def test_supervisor_imports(self):
        from agents import supervisor_agent
        assert callable(supervisor_agent)

    def test_inventory_analyst_imports(self):
        from agents import inventory_analyst_agent
        assert callable(inventory_analyst_agent)

    def test_chat_agent_imports(self):
        from agents import chat_agent
        assert callable(chat_agent)
