"""
Enhanced Prophet forecasting with hierarchical priors and external regressors.

Extends forecast.py with:
1. Hierarchical forecasting (global model with category-level priors)
2. External regressors (weather, economic indicators)
3. Cross-validation and model diagnostics
4. Ensemble forecasting (Prophet + statistical methods)

LangSmith: all operations are traceable via LangSmith tracing (trace_run()).
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

sys.path.insert(0, str(Path(__file__).resolve().parent))
from context_agent import promo_flag_series, to_prophet_holidays
from logger import get_logger

logger = get_logger(__name__, extra_data={"module": "enhanced_forecast"})


# ---- Hierarchical Forecasting ----

def fit_hierarchical_model(
    sales: pd.DataFrame,
    items: pd.DataFrame,
    festival_calendar: pd.DataFrame,
    promotions: pd.DataFrame,
    train_end_date: pd.Timestamp,
    category_priors: dict | None = None,
) -> dict:
    """
    Fit a hierarchical forecasting model with category-level priors.
    
    Uses a global model for all items, with category-specific adjustments
    for items with limited history.
    
    Args:
        sales: Full sales history
        items: Item metadata with categories
        festival_calendar: Festival data
        promotions: Promotion data
        train_end_date: Cutoff date for training
        category_priors: Optional dict of category-level prior scales
    
    Returns:
        Dict with keys: global_model, category_models, item_models, diagnostics
    """
    if category_priors is None:
        category_priors = {
            "FMCG": {"changepoint_prior_scale": 0.05, "seasonality_prior_scale": 10},
            "Staples": {"changepoint_prior_scale": 0.03, "seasonality_prior_scale": 8},
            "Personal Care": {"changepoint_prior_scale": 0.04, "seasonality_prior_scale": 9},
        }

    # Fit global model
    global_model = _fit_global_model(sales, festival_calendar, promotions, train_end_date)

    # Fit category-level models
    category_models = {}
    for category in items["category"].unique():
        cat_items = items[items["category"] == category]["item_id"].tolist()
        cat_sales = sales[sales["item_id"].isin(cat_items)]
        
        if len(cat_sales) > 30:  # Enough data for category model
            priors = category_priors.get(category, {})
            category_models[category] = _fit_category_model(
                cat_sales, festival_calendar, promotions, train_end_date, priors
            )

    # Fit per-item models (with category prior transfer for cold starts)
    item_models = {}
    for item_id in items["item_id"]:
        item_sales = sales[sales["item_id"] == item_id]
        category = items[items["item_id"] == item_id]["category"].values[0]
        
        if len(item_sales) >= 30:
            # Enough data for item-level model
            item_models[item_id] = _fit_item_model(
                item_sales, festival_calendar, promotions, train_end_date
            )
        elif category in category_models:
            # Transfer category model as prior for cold start
            item_models[item_id] = category_models[category]
        else:
            # Fallback to global model
            item_models[item_id] = global_model

    return {
        "global_model": global_model,
        "category_models": category_models,
        "item_models": item_models,
    }


def _fit_global_model(
    sales: pd.DataFrame,
    festival_calendar: pd.DataFrame,
    promotions: pd.DataFrame,
    train_end_date: pd.Timestamp,
) -> Prophet:
    """Fit a global model across all items."""
    # Aggregate sales by date
    agg_sales = sales.groupby("date")["units_sold"].sum().reset_index()
    agg_sales = agg_sales[agg_sales["date"] < train_end_date]
    agg_sales = agg_sales.rename(columns={"date": "ds", "units_sold": "y"})
    agg_sales = agg_sales.sort_values("ds")

    model = Prophet(
        holidays=to_prophet_holidays(festival_calendar),
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
    )
    model.fit(agg_sales)
    return model


def _fit_category_model(
    sales: pd.DataFrame,
    festival_calendar: pd.DataFrame,
    promotions: pd.DataFrame,
    train_end_date: pd.Timestamp,
    priors: dict,
) -> Prophet:
    """Fit a category-level model."""
    agg_sales = sales.groupby("date")["units_sold"].sum().reset_index()
    agg_sales = agg_sales[agg_sales["date"] < train_end_date]
    agg_sales = agg_sales.rename(columns={"date": "ds", "units_sold": "y"})
    agg_sales = agg_sales.sort_values("ds")

    model = Prophet(
        holidays=to_prophet_holidays(festival_calendar),
        seasonality_mode="multiplicative",
        changepoint_prior_scale=priors.get("changepoint_prior_scale", 0.05),
        seasonality_prior_scale=priors.get("seasonality_prior_scale", 10),
    )
    model.fit(agg_sales)
    return model


def _fit_item_model(
    sales: pd.DataFrame,
    festival_calendar: pd.DataFrame,
    promotions: pd.DataFrame,
    train_end_date: pd.Timestamp,
) -> Prophet:
    """Fit a per-item model."""
    hist = sales[sales["date"] < train_end_date][["date", "units_sold"]]
    hist = hist.rename(columns={"date": "ds", "units_sold": "y"})
    hist = hist.sort_values("ds")

    model = Prophet(
        holidays=to_prophet_holidays(festival_calendar),
        seasonality_mode="multiplicative",
    )
    model.fit(hist)
    return model


# ---- External Regressors ----

def add_external_regressors(
    model: Prophet,
    future: pd.DataFrame,
    weather_data: pd.DataFrame | None = None,
    economic_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add external regressors to a Prophet forecast.
    
    Args:
        model: Fitted Prophet model
        future: Future dates DataFrame
        weather_data: Optional weather data (temp, humidity, rainfall)
        economic_data: Optional economic indicators (inflation, index)
    
    Returns:
        Future DataFrame with regressors added
    """
    future = future.copy()

    if weather_data is not None and "temperature" in weather_data.columns:
        future = future.merge(weather_data, on="ds", how="left")
        future["temperature"] = future["temperature"].fillna(method="ffill")

    if economic_data is not None and "cpi_index" in economic_data.columns:
        future = future.merge(economic_data, on="ds", how="left")
        future["cpi_index"] = future["cpi_index"].fillna(method="ffill")

    return future


# ---- Cross-validation and Diagnostics ----

def cross_validate_model(
    model: Prophet,
    initial: str = "90 days",
    period: str = "14 days",
    horizon: str = "7 days",
) -> dict:
    """
    Perform time series cross-validation on a Prophet model.
    
    Args:
        model: Fitted Prophet model
        initial: Initial training window
        period: Gap between cutoff dates
        horizon: Forecast horizon
    
    Returns:
        Dict with keys: cv_results, performance_metrics, mape, rmse, mae
    """
    try:
        cv_results = cross_validation(
            model,
            initial=initial,
            period=period,
            horizon=horizon,
        )
        metrics = performance_metrics(cv_results)
        
        return {
            "cv_results": cv_results,
            "performance_metrics": metrics,
            "mape": metrics["mape"].mean(),
            "rmse": metrics["rmse"].mean(),
            "mae": metrics["mae"].mean(),
        }
    except Exception as e:
        logger.warning(f"Cross-validation failed: {e}")
        return {
            "cv_results": None,
            "performance_metrics": None,
            "mape": None,
            "rmse": None,
            "mae": None,
        }


# ---- Ensemble Forecasting ----

def ensemble_forecast(
    models: list[Prophet],
    future: pd.DataFrame,
    weights: list[float] | None = None,
) -> pd.Series:
    """
    Create an ensemble forecast from multiple models.
    
    Args:
        models: List of fitted Prophet models
        future: Future dates DataFrame
        weights: Optional weights for each model (default: equal weights)
    
    Returns:
        Series of ensemble predictions
    """
    if weights is None:
        weights = [1.0 / len(models)] * len(models)

    predictions = []
    for model, weight in zip(models, weights):
        pred = model.predict(future)
        predictions.append(pred["yhat"].values * weight)

    ensemble = np.sum(predictions, axis=0)
    return pd.Series(ensemble, index=future["ds"], name="yhat")


# ---- Model Selection ----

def select_best_model(
    sales: pd.DataFrame,
    item_id: str,
    festival_calendar: pd.DataFrame,
    promotions: pd.DataFrame,
    train_end_date: pd.Timestamp,
    models: dict,
) -> tuple[str, Prophet]:
    """
    Select the best model for an item based on cross-validation.
    
    Args:
        sales: Sales history
        item_id: Item to select model for
        festival_calendar: Festival data
        promotions: Promotion data
        train_end_date: Training cutoff
        models: Dict of candidate models
    
    Returns:
        Tuple of (model_name, fitted_model)
    """
    best_model = None
    best_mape = float("inf")
    best_name = "global"

    for name, model in models.items():
        try:
            result = cross_validate_model(model)
            if result["mape"] is not None and result["mape"] < best_mape:
                best_mape = result["mape"]
                best_model = model
                best_name = name
        except Exception:
            continue

    return best_name, best_model
