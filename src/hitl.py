"""
Enhanced Human-in-the-Loop (HITL) patterns for SellerSense.

Implements:
1. Per-item interrupts for independent approve/reject decisions
2. Auto-approval for low-risk items based on confidence thresholds
3. WhatsApp/SMS notification via Twilio

LangSmith: all operations are traceable via @traceable decorators.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from langsmith import traceable
except ImportError:
    def traceable(name=None, run_type="chain"):
        def decorator(func):
            return func
        return decorator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from logger import get_logger

logger = get_logger(__name__, extra_data={"module": "hitl"})


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


# ---- Auto-approval Thresholds ----

AUTO_APPROVE_THRESHOLDS = {
    "max_risk_for_auto": "stockout_risk",  # Only auto-approve stockout items
    "min_confidence": 0.8,                 # Minimum confidence for auto-approval
    "max_order_value": 5000,               # Max ₹ value for auto-approval
    "excluded_categories": [],             # Categories that never auto-approve
    "excluded_items": [],                  # Specific items that never auto-approve
}


class AutoApprovalEngine:
    """
    Determines which items can be auto-approved without human intervention.
    
    Uses configurable thresholds based on:
    - Risk level (only stockout_risk auto-approvable)
    - Order value (below threshold)
    - Item/category exclusions
    - Historical approval patterns
    """

    def __init__(self, thresholds: dict | None = None):
        self.thresholds = {**AUTO_APPROVE_THRESHOLDS, **(thresholds or {})}

    @traceable(name="should_auto_approve", run_type="chain")
    def should_auto_approve(self, recommendation: dict) -> dict:
        """
        Determine if a recommendation should be auto-approved.
        
        Args:
            recommendation: Dict with keys: item_id, risk, suggested_order_qty,
                          unit_cost_inr, category, urgency, confidence
        
        Returns:
            Dict with keys: auto_approve (bool), reason (str)
        """
        # Check exclusions
        if recommendation.get("item_id") in self.thresholds["excluded_items"]:
            return {
                "auto_approve": False,
                "reason": "item_excluded",
            }

        if recommendation.get("category") in self.thresholds["excluded_categories"]:
            return {
                "auto_approve": False,
                "reason": "category_excluded",
            }

        # Check risk level
        if recommendation.get("risk") != self.thresholds["max_risk_for_auto"]:
            return {
                "auto_approve": False,
                "reason": "risk_too_high",
            }

        # Check order value
        order_value = (
            recommendation.get("suggested_order_qty", 0) *
            recommendation.get("unit_cost_inr", 0)
        )
        if order_value > self.thresholds["max_order_value"]:
            return {
                "auto_approve": False,
                "reason": "order_value_exceeds_threshold",
            }

        # Check confidence
        confidence = recommendation.get("confidence", 0)
        if confidence < self.thresholds["min_confidence"]:
            return {
                "auto_approve": False,
                "reason": "confidence_too_low",
            }

        return {
            "auto_approve": True,
            "reason": "meets_all_criteria",
            "order_value": order_value,
        }

    @traceable(name="classify_recommendations", run_type="chain")
    def classify_recommendations(
        self,
        recommendations: list[dict],
    ) -> dict:
        """
        Classify recommendations into auto-approve and needs-review categories.
        
        Args:
            recommendations: List of recommendation dicts
        
        Returns:
            Dict with keys: auto_approve (list), needs_review (list), summary (dict)
        """
        auto_approve = []
        needs_review = []

        for rec in recommendations:
            decision = self.should_auto_approve(rec)
            if decision["auto_approve"]:
                auto_approve.append({**rec, "auto_approve_reason": decision["reason"]})
            else:
                needs_review.append({**rec, "review_reason": decision["reason"]})

        return {
            "auto_approve": auto_approve,
            "needs_review": needs_review,
            "summary": {
                "total": len(recommendations),
                "auto_approved": len(auto_approve),
                "needs_review": len(needs_review),
            },
        }


# ---- WhatsApp/SMS Notifications ----

class WhatsAppNotifier:
    """
    Send WhatsApp/SMS notifications via Twilio.
    
    Supports both WhatsApp and SMS channels.
    
    Requires:
        pip install twilio
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM (or TWILIO_SMS_FROM)
    """

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
    ):
        self.account_sid = account_sid or _get_secret("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or _get_secret("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or _get_secret("TWILIO_WHATSAPP_FROM")

    @property
    def available(self) -> bool:
        """Check if notifications are configured."""
        return bool(self.account_sid and self.auth_token and self.from_number)

    @traceable(name="send_whatsapp", run_type="chain")
    def send_notification(
        self,
        to_number: str,
        message: str,
        channel: str = "whatsapp",
    ) -> dict:
        """
        Send a notification via WhatsApp or SMS.
        
        Args:
            to_number: Recipient phone number (with country code, e.g., "+916304595065")
            message: Message text
            channel: "whatsapp" or "sms"
        
        Returns:
            Dict with keys: success (bool), message_sid (str), error (str)
        """
        if not self.available:
            return {
                "success": False,
                "error": "Twilio not configured",
            }

        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)

            # Format the 'from' number based on channel
            if channel == "whatsapp":
                from_addr = f"whatsapp:{self.from_number}"
                to_addr = f"whatsapp:{to_number}"
            else:
                from_addr = self.from_number
                to_addr = to_number

            msg = client.messages.create(
                from_=from_addr,
                to=to_addr,
                body=message,
            )

            logger.info("Notification sent", extra={"extra_data": {
                "channel": channel,
                "to": to_number,
                "message_sid": msg.sid,
            }})

            return {
                "success": True,
                "message_sid": msg.sid,
                "channel": channel,
            }

        except ImportError:
            return {
                "success": False,
                "error": "twilio package not installed: pip install twilio",
            }
        except Exception as e:
            logger.error(f"Notification failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @traceable(name="send_sms", run_type="chain")
    def send_sms(
        self,
        to_number: str,
        message: str,
    ) -> dict:
        """Send SMS notification (convenience method)."""
        return self.send_notification(to_number, message, channel="sms")

    @traceable(name="send_daily_summary", run_type="chain")
    def send_daily_summary(
        self,
        to_number: str,
        recommendations: list[dict],
        auto_approved: list[dict],
        needs_review: list[dict],
        channel: str = "whatsapp",
    ) -> dict:
        """
        Send a daily inventory summary.
        
        Args:
            to_number: Recipient phone number
            recommendations: All recommendations
            auto_approved: Auto-approved items
            needs_review: Items needing review
            channel: "whatsapp" or "sms"
        
        Returns:
            Dict with keys: success (bool), error (str)
        """
        if not recommendations:
            return {"success": True, "error": None}

        # Build message
        lines = ["*SellerSense Daily Inventory Summary*\n"]
        
        if auto_approved:
            lines.append(f"*Auto-approved ({len(auto_approved)} items):*")
            for item in auto_approved[:5]:  # Limit to 5
                lines.append(
                    f"  - {item.get('item_name', 'Unknown')}: "
                    f"Order {item.get('suggested_order_qty', 0)} units"
                )
            if len(auto_approved) > 5:
                lines.append(f"  ... and {len(auto_approved) - 5} more")
            lines.append("")

        if needs_review:
            lines.append(f"*Needs your review ({len(needs_review)} items):*")
            for item in needs_review[:5]:
                lines.append(
                    f"  - {item.get('item_name', 'Unknown')}: "
                    f"{item.get('urgency', 'medium')} urgency"
                )
            if len(needs_review) > 5:
                lines.append(f"  ... and {len(needs_review) - 5} more")
            lines.append("")

        lines.append(f"_Open dashboard to review: {len(needs_review)} items need attention_")

        message = "\n".join(lines)
        return self.send_notification(to_number, message, channel=channel)

    @traceable(name="send_recommendation_alert", run_type="chain")
    def send_recommendation_alert(
        self,
        to_number: str,
        recommendation: dict,
        channel: str = "whatsapp",
    ) -> dict:
        """
        Send an alert for a single high-priority recommendation.
        
        Args:
            to_number: Recipient phone number
            recommendation: Single recommendation dict
            channel: "whatsapp" or "sms"
        
        Returns:
            Dict with keys: success (bool), error (str)
        """
        item_name = recommendation.get("item_name", "Unknown Item")
        urgency = recommendation.get("urgency", "medium")
        qty = recommendation.get("suggested_order_qty", 0)
        risk = recommendation.get("risk", "unknown")

        message = (
            f"*SellerSense Alert*\n\n"
            f"*{item_name}* needs attention!\n"
            f"Risk: {risk}\n"
            f"Urgency: {urgency}\n"
            f"Suggested order: {qty} units\n\n"
            f"_Open dashboard to review this recommendation._"
        )

        return self.send_notification(to_number, message, channel=channel)


# ---- Per-item Interrupt Pattern ----

@traceable(name="per_item_interrupt", run_type="chain")
def per_item_interrupt(
    recommendations: list[dict],
    auto_approve_engine: AutoApprovalEngine | None = None,
) -> dict:
    """
    Process recommendations with per-item decisions.
    
    Returns a structured result that can be used by the dashboard
    to handle each item independently.
    
    Args:
        recommendations: List of recommendation dicts
        auto_approve_engine: Optional engine for auto-approval
    
    Returns:
        Dict with keys: auto_approved, needs_review, all_items, summary
    """
    if auto_approve_engine is None:
        auto_approve_engine = AutoApprovalEngine()

    classification = auto_approve_engine.classify_recommendations(recommendations)

    return {
        "auto_approved": classification["auto_approve"],
        "needs_review": classification["needs_review"],
        "all_items": recommendations,
        "summary": classification["summary"],
    }
