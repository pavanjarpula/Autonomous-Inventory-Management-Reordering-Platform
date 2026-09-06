"""
Google Calendar integration for dynamic festival detection.

Replaces static festival_calendar.csv with live calendar data from
Google Calendar API. Falls back to CSV when credentials are unavailable.

Requires:
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib

Environment:
    GOOGLE_CALENDAR_ID: Calendar ID (default: "primary")
    GOOGLE_CREDENTIALS_PATH: Path to service account JSON or OAuth token

LangSmith: all operations are traceable via @traceable decorators.
"""

import json
import sys
from datetime import datetime, timedelta
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

logger = get_logger(__name__, extra_data={"module": "google_calendar"})


# Festival-to-category mapping for the 25-SKU dataset
FESTIVAL_CATEGORY_MAP = {
    "Diwali": ["FMCG", "Staples", "Personal Care"],
    "Eid": ["FMCG", "Staples"],
    "Holi": ["FMCG", "Personal Care"],
    "Navratri": ["FMCG", "Staples"],
    "Raksha Bandhan": ["FMCG", "Personal Care"],
    "Christmas": ["FMCG", "Personal Care"],
    "Pongal": ["Staples", "FMCG"],
    "Onam": ["FMCG", "Staples"],
    "Dussehra": ["FMCG", "Staples"],
    "Janmashtami": ["FMCG", "Staples"],
    "Ganesh Chaturthi": ["FMCG", "Staples"],
    "Independence Day": ["FMCG"],
    "Republic Day": ["FMCG"],
}

DEFAULT_UPLIFT = 1.4
DEFAULT_RAMP_DAYS = 7


class GoogleCalendarClient:
    """
    Client for fetching festival/event data from Google Calendar.
    
    Falls back gracefully when credentials are not available.
    """

    def __init__(
        self,
        calendar_id: str = "primary",
        credentials_path: str | None = None,
    ):
        self.calendar_id = calendar_id
        self.credentials_path = credentials_path or self._find_credentials()
        self._service = None

    def _find_credentials(self) -> str | None:
        """Look for credentials in common locations."""
        import os
        
        # Check environment variable
        creds_path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
        if creds_path and Path(creds_path).exists():
            return creds_path

        # Check common locations
        common_paths = [
            Path.home() / ".config" / "gcloud" / "credentials.json",
            Path("credentials.json"),
            Path("service_account.json"),
        ]
        for path in common_paths:
            if path.exists():
                return str(path)

        return None

    @property
    def available(self) -> bool:
        """Check if Google Calendar integration is available."""
        try:
            from google.oauth2 import service_account
            from googleapiclient.discovery import build
            return self.credentials_path is not None
        except ImportError:
            return False

    def _get_service(self):
        """Get Google Calendar API service."""
        if self._service is not None:
            return self._service

        if not self.available:
            raise RuntimeError(
                "Google Calendar integration requires: "
                "pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib"
            )

        from google.oauth2 import service_account
        from googleapiclient.discovery import build

        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=["https://www.googleapis.com/auth/calendar.readonly"],
        )
        self._service = build("calendar", "v3", credentials=credentials)
        return self._service

    @traceable(name="fetch_calendar_events", run_type="chain")
    def fetch_events(
        self,
        time_min: datetime | None = None,
        time_max: datetime | None = None,
        max_results: int = 100,
    ) -> list[dict]:
        """
        Fetch events from Google Calendar.
        
        Args:
            time_min: Start of time range (default: now)
            time_max: End of time range (default: 90 days from now)
            max_results: Maximum events to fetch
        
        Returns:
            List of event dicts with keys: summary, start, end, description
        """
        if not self.available:
            logger.warning("Google Calendar not available, returning empty events")
            return []

        service = self._get_service()

        if time_min is None:
            time_min = datetime.utcnow()
        if time_max is None:
            time_max = time_min + timedelta(days=90)

        try:
            events_result = service.events().list(
                calendarId=self.calendar_id,
                timeMin=time_min.isoformat() + "Z",
                timeMax=time_max.isoformat() + "Z",
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            ).execute()

            events = events_result.get("items", [])
            logger.info("Fetched calendar events", extra={"extra_data": {
                "n_events": len(events),
            }})
            return events

        except Exception as e:
            logger.error(f"Failed to fetch calendar events: {e}")
            return []


@traceable(name="events_to_festival_calendar", run_type="chain")
def events_to_festival_calendar(
    events: list[dict],
    category_map: dict | None = None,
) -> pd.DataFrame:
    """
    Convert Google Calendar events to festival_calendar.csv format.
    
    Args:
        events: List of Google Calendar event dicts
        category_map: Festival-to-category mapping (default: FESTIVAL_CATEGORY_MAP)
    
    Returns:
        DataFrame with columns: date, festival_name, affected_categories,
                                 uplift_multiplier, ramp_days_before
    """
    if category_map is None:
        category_map = FESTIVAL_CATEGORY_MAP

    rows = []
    for event in events:
        summary = event.get("summary", "")
        start = event.get("start", {})
        
        # Parse date
        if "date" in start:
            date_str = start["date"]
        elif "dateTime" in start:
            date_str = start["dateTime"][:10]
        else:
            continue

        try:
            date = pd.Timestamp(date_str)
        except Exception:
            continue

        # Find matching category
        affected_categories = "FMCG"  # default
        for festival_name, categories in category_map.items():
            if festival_name.lower() in summary.lower():
                affected_categories = ", ".join(categories)
                break

        rows.append({
            "date": date,
            "festival_name": summary,
            "affected_categories": affected_categories,
            "uplift_multiplier": DEFAULT_UPLIFT,
            "ramp_days_before": DEFAULT_RAMP_DAYS,
        })

    if not rows:
        return pd.DataFrame(columns=[
            "date", "festival_name", "affected_categories",
            "uplift_multiplier", "ramp_days_before"
        ])

    return pd.DataFrame(rows)


@traceable(name="load_festival_data", run_type="chain")
def load_festival_data(
    csv_path: str | Path | None = None,
    calendar_id: str = "primary",
    credentials_path: str | None = None,
    use_google_calendar: bool = True,
) -> pd.DataFrame:
    """
    Load festival data, preferring Google Calendar when available.
    
    Priority:
    1. Google Calendar API (if credentials available and use_google_calendar=True)
    2. Static CSV file (fallback)
    
    Args:
        csv_path: Path to fallback CSV file
        calendar_id: Google Calendar ID
        credentials_path: Path to credentials JSON
        use_google_calendar: Whether to try Google Calendar first
    
    Returns:
        DataFrame with festival calendar data
    """
    # Try Google Calendar first
    if use_google_calendar:
        try:
            client = GoogleCalendarClient(calendar_id, credentials_path)
            if client.available:
                events = client.fetch_events()
                if events:
                    return events_to_festival_calendar(events)
                logger.info("No events found in Google Calendar, falling back to CSV")
            else:
                logger.info("Google Calendar not configured, using CSV fallback")
        except Exception as e:
            logger.warning(f"Google Calendar failed: {e}, using CSV fallback")

    # Fallback to CSV
    if csv_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "festival_calendar.csv"
    
    csv_path = Path(csv_path)
    if csv_path.exists():
        return pd.read_csv(csv_path, parse_dates=["date"])
    
    logger.warning("No festival data available")
    return pd.DataFrame(columns=[
        "date", "festival_name", "affected_categories",
        "uplift_multiplier", "ramp_days_before"
    ])
