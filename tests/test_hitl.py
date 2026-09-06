"""
Tests for src/hitl.py (Phase 5: Enhanced HITL).

Tests cover:
- Auto-approval engine logic
- WhatsApp notification setup
- Per-item interrupt pattern
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestAutoApprovalEngine:
    """Tests for AutoApprovalEngine."""

    def test_import_module(self):
        from hitl import AutoApprovalEngine, WhatsAppNotifier, per_item_interrupt
        assert AutoApprovalEngine is not None
        assert WhatsAppNotifier is not None
        assert per_item_interrupt is not None

    def test_auto_approve_low_risk_low_value(self):
        from hitl import AutoApprovalEngine
        engine = AutoApprovalEngine()
        
        rec = {
            "item_id": "I001",
            "risk": "stockout_risk",
            "suggested_order_qty": 10,
            "unit_cost_inr": 50,  # ₹500 total
            "category": "FMCG",
            "urgency": "high",
            "confidence": 0.9,
        }
        
        result = engine.should_auto_approve(rec)
        assert result["auto_approve"] is True

    def test_reject_high_risk(self):
        from hitl import AutoApprovalEngine
        engine = AutoApprovalEngine()
        
        rec = {
            "item_id": "I001",
            "risk": "overstock",
            "suggested_order_qty": 10,
            "unit_cost_inr": 50,
            "category": "FMCG",
            "urgency": "medium",
            "confidence": 0.9,
        }
        
        result = engine.should_auto_approve(rec)
        assert result["auto_approve"] is False
        assert result["reason"] == "risk_too_high"

    def test_reject_high_value(self):
        from hitl import AutoApprovalEngine
        engine = AutoApprovalEngine()
        
        rec = {
            "item_id": "I001",
            "risk": "stockout_risk",
            "suggested_order_qty": 100,
            "unit_cost_inr": 100,  # ₹10,000 total
            "category": "FMCG",
            "urgency": "high",
            "confidence": 0.9,
        }
        
        result = engine.should_auto_approve(rec)
        assert result["auto_approve"] is False
        assert result["reason"] == "order_value_exceeds_threshold"

    def test_reject_excluded_item(self):
        from hitl import AutoApprovalEngine
        engine = AutoApprovalEngine({"excluded_items": ["I001"]})
        
        rec = {
            "item_id": "I001",
            "risk": "stockout_risk",
            "suggested_order_qty": 10,
            "unit_cost_inr": 50,
            "category": "FMCG",
            "urgency": "high",
            "confidence": 0.9,
        }
        
        result = engine.should_auto_approve(rec)
        assert result["auto_approve"] is False
        assert result["reason"] == "item_excluded"

    def test_reject_low_confidence(self):
        from hitl import AutoApprovalEngine
        engine = AutoApprovalEngine()
        
        rec = {
            "item_id": "I001",
            "risk": "stockout_risk",
            "suggested_order_qty": 10,
            "unit_cost_inr": 50,
            "category": "FMCG",
            "urgency": "high",
            "confidence": 0.5,  # Below 0.8 threshold
        }
        
        result = engine.should_auto_approve(rec)
        assert result["auto_approve"] is False
        assert result["reason"] == "confidence_too_low"

    def test_classify_recommendations(self):
        from hitl import AutoApprovalEngine
        engine = AutoApprovalEngine()
        
        recs = [
            {
                "item_id": "I001",
                "risk": "stockout_risk",
                "suggested_order_qty": 10,
                "unit_cost_inr": 50,
                "category": "FMCG",
                "confidence": 0.9,
            },
            {
                "item_id": "I002",
                "risk": "overstock",
                "suggested_order_qty": 5,
                "unit_cost_inr": 100,
                "category": "FMCG",
                "confidence": 0.8,
            },
        ]
        
        result = engine.classify_recommendations(recs)
        assert result["summary"]["total"] == 2
        assert result["summary"]["auto_approved"] == 1
        assert result["summary"]["needs_review"] == 1


class TestWhatsAppNotifier:
    """Tests for WhatsAppNotifier."""

    def test_notifier_not_available_without_config(self):
        from hitl import WhatsAppNotifier
        notifier = WhatsAppNotifier()
        assert notifier.available is False

    def test_send_notification_fails_gracefully(self):
        from hitl import WhatsAppNotifier
        notifier = WhatsAppNotifier()
        
        result = notifier.send_notification("+1234567890", "Test")
        assert result["success"] is False
        assert "not configured" in result["error"]


class TestPerItemInterrupt:
    """Tests for per_item_interrupt."""

    def test_per_item_interrupt_returns_structure(self):
        from hitl import per_item_interrupt
        recs = [
            {
                "item_id": "I001",
                "risk": "stockout_risk",
                "suggested_order_qty": 10,
                "unit_cost_inr": 50,
                "category": "FMCG",
                "confidence": 0.9,
            },
        ]
        
        result = per_item_interrupt(recs)
        assert "auto_approved" in result
        assert "needs_review" in result
        assert "all_items" in result
        assert "summary" in result

    def test_per_item_interrupt_empty(self):
        from hitl import per_item_interrupt
        result = per_item_interrupt([])
        assert result["summary"]["total"] == 0
        assert result["summary"]["auto_approved"] == 0
