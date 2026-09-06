"""
Tests for src/google_calendar.py (Phase 4: Google Calendar integration).

Tests cover:
- Calendar client initialization
- Event-to-calendar conversion
- Fallback to CSV when credentials unavailable
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestGoogleCalendar:
    """Tests for Google Calendar integration."""

    def test_import_module(self):
        from google_calendar import GoogleCalendarClient, load_festival_data
        assert GoogleCalendarClient is not None
        assert load_festival_data is not None

    def test_client_creation_without_credentials(self):
        from google_calendar import GoogleCalendarClient
        client = GoogleCalendarClient()
        assert client.calendar_id == "primary"
        assert client.available is False

    def test_events_to_festival_calendar(self):
        from google_calendar import events_to_festival_calendar
        events = [
            {
                "summary": "Diwali Celebration",
                "start": {"date": "2026-10-20"},
                "end": {"date": "2026-10-21"},
            },
            {
                "summary": "Christmas Day",
                "start": {"date": "2026-12-25"},
                "end": {"date": "2026-12-26"},
            },
        ]
        df = events_to_festival_calendar(events)
        assert len(df) == 2
        assert "date" in df.columns
        assert "festival_name" in df.columns
        assert "affected_categories" in df.columns
        assert "Diwali Celebration" in df["festival_name"].values

    def test_events_to_festival_calendar_empty(self):
        from google_calendar import events_to_festival_calendar
        df = events_to_festival_calendar([])
        assert len(df) == 0

    def test_events_to_festival_calendar_unknown_festival(self):
        from google_calendar import events_to_festival_calendar
        events = [
            {
                "summary": "Unknown Festival XYZ",
                "start": {"date": "2026-11-01"},
            },
        ]
        df = events_to_festival_calendar(events)
        assert len(df) == 1
        # Should default to FMCG
        assert df.iloc[0]["affected_categories"] == "FMCG"

    def test_load_festival_data_csv_fallback(self):
        from google_calendar import load_festival_data
        # Should fall back to CSV since no Google credentials
        df = load_festival_data(use_google_calendar=False)
        assert isinstance(df, pd.DataFrame)
        # CSV exists in data directory
        assert len(df) > 0 or "festival_name" in df.columns

    def test_load_festival_data_with_csv_path(self):
        from google_calendar import load_festival_data
        csv_path = Path(__file__).resolve().parent.parent / "data" / "festival_calendar.csv"
        df = load_festival_data(csv_path=csv_path, use_google_calendar=False)
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_load_festival_data_google_unavailable(self):
        from google_calendar import load_festival_data
        # Google Calendar not configured, should fall back to CSV
        df = load_festival_data(use_google_calendar=True)
        assert isinstance(df, pd.DataFrame)
