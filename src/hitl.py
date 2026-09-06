"""
Enhanced Human-in-the-Loop (HITL) patterns for SellerSense.

Implements:
1. Per-item interrupts for independent approve/reject decisions
2. Auto-approval for low-risk items based on confidence thresholds
3. WhatsApp/Email notification via Twilio

LangSmith: all operations are traceable via @traceable decorators.
"""

import json
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


# ---- WhatsApp/Email Notifications ----

class WhatsAppNotifier:
    """
    Send WhatsApp/Email notifications via Twilio.
    
    Supports WhatsApp and Email channels.
    
    Requires:
        pip install twilio requests
        TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM
        TWILIO_EMAIL_FROM (for email notifications)
    """

    def __init__(
        self,
        account_sid: str | None = None,
        auth_token: str | None = None,
        from_number: str | None = None,
        from_email: str | None = None,
    ):
        self.account_sid = account_sid or _get_secret("TWILIO_ACCOUNT_SID")
        self.auth_token = auth_token or _get_secret("TWILIO_AUTH_TOKEN")
        self.from_number = from_number or _get_secret("TWILIO_WHATSAPP_FROM")
        self.from_email = from_email or _get_secret("TWILIO_EMAIL_FROM")

    @property
    def available(self) -> bool:
        """Check if notifications are configured."""
        return bool(self.account_sid and self.auth_token)

    @property
    def whatsapp_available(self) -> bool:
        """Check if WhatsApp is configured."""
        return bool(self.account_sid and self.auth_token and self.from_number)

    @property
    def email_available(self) -> bool:
        """Check if email is configured."""
        return bool(self.account_sid and self.auth_token and self.from_email)

    @traceable(name="send_whatsapp", run_type="chain")
    def send_notification(
        self,
        to_number: str,
        message: str,
        channel: str = "whatsapp",
    ) -> dict:
        """
        Send a notification via WhatsApp.
        
        Args:
            to_number: Recipient phone number (with country code, e.g., "+916304595065")
            message: Message text
            channel: "whatsapp"
        
        Returns:
            Dict with keys: success (bool), message_sid (str), error (str)
        """
        if not self.whatsapp_available:
            return {
                "success": False,
                "error": "WhatsApp not configured",
            }

        try:
            from twilio.rest import Client
            client = Client(self.account_sid, self.auth_token)

            from_addr = f"whatsapp:{self.from_number}"
            to_addr = f"whatsapp:{to_number}"

            msg = client.messages.create(
                from_=from_addr,
                to=to_addr,
                body=message,
            )

            logger.info("WhatsApp sent", extra={"extra_data": {
                "to": to_number,
                "message_sid": msg.sid,
            }})

            return {
                "success": True,
                "message_sid": msg.sid,
                "channel": "whatsapp",
            }

        except ImportError:
            return {
                "success": False,
                "error": "twilio package not installed: pip install twilio",
            }
        except Exception as e:
            logger.error(f"WhatsApp failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @traceable(name="send_email", run_type="chain")
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
    ) -> dict:
        """
        Send an email via Twilio SendGrid.
        
        Args:
            to_email: Recipient email address
            subject: Email subject
            html_content: HTML email body
        
        Returns:
            Dict with keys: success (bool), message_id (str), error (str)
        """
        if not self.email_available:
            return {
                "success": False,
                "error": "Email not configured",
            }

        try:
            import requests

            url = "https://comms.twilio.com/v1/Emails"
            
            payload = {
                "from": {
                    "address": self.from_email,
                    "name": "SellerSense"
                },
                "to": [
                    {"address": to_email}
                ],
                "content": {
                    "subject": subject,
                    "html": html_content
                }
            }

            response = requests.post(
                url,
                auth=(self.account_sid, self.auth_token),
                headers={"Content-Type": "application/json"},
                json=payload,
            )

            if response.status_code in [200, 201]:
                result = response.json()
                message_id = result.get("sid", "")
                
                logger.info("Email sent", extra={"extra_data": {
                    "to": to_email,
                    "message_id": message_id,
                }})

                return {
                    "success": True,
                    "message_id": message_id,
                    "channel": "email",
                }
            else:
                return {
                    "success": False,
                    "error": f"Twilio API error: {response.status_code} - {response.text}",
                }

        except ImportError:
            return {
                "success": False,
                "error": "requests package not installed: pip install requests",
            }
        except Exception as e:
            logger.error(f"Email failed: {e}")
            return {
                "success": False,
                "error": str(e),
            }

    @traceable(name="send_daily_summary_email", run_type="chain")
    def send_daily_summary_email(
        self,
        to_email: str,
        recommendations: list[dict],
        auto_approved: list[dict],
        needs_review: list[dict],
    ) -> dict:
        """
        Send a daily inventory summary via email.
        
        Args:
            to_email: Recipient email address
            recommendations: All recommendations
            auto_approved: Auto-approved items
            needs_review: Items needing review
        
        Returns:
            Dict with keys: success (bool), error (str)
        """
        if not recommendations:
            return {"success": True, "error": None}

        # Build HTML email
        html_parts = [
            "<!DOCTYPE html>",
            "<html>",
            "<head><style>",
            "body { font-family: Arial, sans-serif; line-height: 1.6; color: #333; }",
            ".container { max-width: 600px; margin: 0 auto; padding: 20px; }",
            ".header { background: #4CAF50; color: white; padding: 20px; text-align: center; }",
            ".section { margin: 20px 0; padding: 15px; background: #f9f9f9; border-radius: 5px; }",
            ".item { padding: 8px 0; border-bottom: 1px solid #eee; }",
            ".high { color: #e74c3c; font-weight: bold; }",
            ".medium { color: #f39c12; }",
            ".low { color: #27ae60; }",
            ".footer { text-align: center; padding: 20px; color: #666; font-size: 12px; }",
            "</style></head>",
            "<body>",
            "<div class='container'>",
            "<div class='header'><h1>SellerSense Daily Summary</h1></div>",
        ]

        # Auto-approved section
        if auto_approved:
            html_parts.append("<div class='section'>")
            html_parts.append(f"<h2>Auto-Approved ({len(auto_approved)} items)</h2>")
            for item in auto_approved[:10]:
                name = item.get("item_name", "Unknown")
                qty = item.get("suggested_order_qty", 0)
                html_parts.append(f"<div class='item'><strong>{name}</strong> - Order {qty} units</div>")
            if len(auto_approved) > 10:
                html_parts.append(f"<div class='item'>... and {len(auto_approved) - 10} more</div>")
            html_parts.append("</div>")

        # Needs review section
        if needs_review:
            html_parts.append("<div class='section'>")
            html_parts.append(f"<h2>Needs Your Review ({len(needs_review)} items)</h2>")
            for item in needs_review[:10]:
                name = item.get("item_name", "Unknown")
                urgency = item.get("urgency", "medium")
                qty = item.get("suggested_order_qty", 0)
                risk = item.get("risk", "unknown")
                html_parts.append(
                    f"<div class='item {urgency}'>"
                    f"<strong>{name}</strong> - {urgency.upper()} urgency<br>"
                    f"Risk: {risk} | Order: {qty} units"
                    f"</div>"
                )
            if len(needs_review) > 10:
                html_parts.append(f"<div class='item'>... and {len(needs_review) - 10} more</div>")
            html_parts.append("</div>")

        # Footer
        html_parts.extend([
            "<div class='footer'>",
            "<p>Open your dashboard to review and approve recommendations.</p>",
            "<p>Best regards,<br><strong>SellerSense Team</strong></p>",
            "</div>",
            "</div>",
            "</body>",
            "</html>",
        ])

        html_content = "\n".join(html_parts)
        subject = f"SellerSense Daily Summary - {len(needs_review)} items need review"

        return self.send_email(to_email, subject, html_content)

    @traceable(name="send_recommendation_alert_email", run_type="chain")
    def send_recommendation_alert_email(
        self,
        to_email: str,
        recommendation: dict,
    ) -> dict:
        """
        Send an alert email for a single high-priority recommendation.
        
        Args:
            to_email: Recipient email address
            recommendation: Single recommendation dict
        
        Returns:
            Dict with keys: success (bool), error (str)
        """
        item_name = recommendation.get("item_name", "Unknown Item")
        urgency = recommendation.get("urgency", "medium")
        qty = recommendation.get("suggested_order_qty", 0)
        risk = recommendation.get("risk", "unknown")

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head><style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .alert {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px; }}
            .high {{ border-left-color: #e74c3c; background: #f8d7da; }}
            .details {{ margin: 15px 0; padding: 10px; background: #f9f9f9; }}
        </style></head>
        <body>
            <div class="alert {'high' if urgency == 'high' else ''}">
                <h2> SellerSense Alert</h2>
                <p><strong>{item_name}</strong> needs your attention!</p>
                <div class="details">
                    <p><strong>Risk:</strong> {risk}</p>
                    <p><strong>Urgency:</strong> {urgency.upper()}</p>
                    <p><strong>Suggested Order:</strong> {qty} units</p>
                </div>
                <p>Open your dashboard to review this recommendation.</p>
                <p><em>Best regards,<br>SellerSense Team</em></p>
            </div>
        </body>
        </html>
        """

        subject = f"SellerSense Alert: {item_name} - {urgency.upper()} urgency"

        return self.send_email(to_email, subject, html_content)


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
