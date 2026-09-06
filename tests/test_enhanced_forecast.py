"""
Tests for src/enhanced_forecast.py (Phase 6: Enhanced Prophet forecasting).

Tests cover:
- Hierarchical model fitting
- External regressors
- Cross-validation
- Ensemble forecasting
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class TestEnhancedForecast:
    """Tests for enhanced forecasting module."""

    def test_import_module(self):
        from enhanced_forecast import (
            fit_hierarchical_model,
            ensemble_forecast,
            cross_validate_model,
        )
        assert fit_hierarchical_model is not None
        assert ensemble_forecast is not None
        assert cross_validate_model is not None

    def test_fit_hierarchical_model(self):
        from enhanced_forecast import fit_hierarchical_model
        data_dir = Path(__file__).resolve().parent.parent / "data"
        
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        items = pd.read_csv(data_dir / "items.csv")
        festival_calendar = pd.read_csv(data_dir / "festival_calendar.csv", parse_dates=["date"])
        promotions = pd.read_csv(data_dir / "promotions.csv", parse_dates=["date"])
        
        train_end = sales["date"].max() - pd.Timedelta(days=14)
        
        result = fit_hierarchical_model(
            sales, items, festival_calendar, promotions, train_end
        )
        
        assert "global_model" in result
        assert "category_models" in result
        assert "item_models" in result
        assert len(result["item_models"]) == len(items)

    def test_ensemble_forecast(self):
        from enhanced_forecast import ensemble_forecast, fit_hierarchical_model
        from prophet import Prophet
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        items = pd.read_csv(data_dir / "items.csv")
        festival_calendar = pd.read_csv(data_dir / "festival_calendar.csv", parse_dates=["date"])
        promotions = pd.read_csv(data_dir / "promotions.csv", parse_dates=["date"])
        
        train_end = sales["date"].max() - pd.Timedelta(days=14)
        result = fit_hierarchical_model(
            sales, items, festival_calendar, promotions, train_end
        )
        
        # Create future dates
        future_dates = pd.date_range(train_end + pd.Timedelta(days=1), periods=7)
        future = pd.DataFrame({"ds": future_dates})
        
        # Ensemble from global + category models
        models = [result["global_model"]] + list(result["category_models"].values())
        ensemble = ensemble_forecast(models[:3], future)  # Limit for speed
        
        assert len(ensemble) == 7
        assert all(v >= 0 for v in ensemble.values)

    def test_cross_validate_model(self):
        from enhanced_forecast import cross_validate_model, fit_hierarchical_model
        
        data_dir = Path(__file__).resolve().parent.parent / "data"
        sales = pd.read_csv(data_dir / "daily_sales.csv", parse_dates=["date"])
        items = pd.read_csv(data_dir / "items.csv")
        festival_calendar = pd.read_csv(data_dir / "festival_calendar.csv", parse_dates=["date"])
        promotions = pd.read_csv(data_dir / "promotions.csv", parse_dates=["date"])
        
        train_end = sales["date"].max() - pd.Timedelta(days=14)
        result = fit_hierarchical_model(
            sales, items, festival_calendar, promotions, train_end
        )
        
        # Cross-validate global model
        cv_result = cross_validate_model(
            result["global_model"],
            initial="60 days",
            period="7 days",
            horizon="3 days",
        )
        
        assert "cv_results" in cv_result
        assert "mape" in cv_result
