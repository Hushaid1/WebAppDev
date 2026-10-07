"""Climate API routes.

Endpoints: /flood-risk, /compound-risk, /alerts, /lagdo-dam.
All flood-risk predictions are served directly from the trained XGBoost model
and static parquet files — no database required.
"""

import os
import json
import time
import logging
import math
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query, UploadFile, File

from .schemas import (
    FloodRiskListResponse,
    FloodRiskResponse,
    CompoundRiskRequest,
    CompoundRiskListResponse,
    CompoundRiskResponse,
    ClimateAlertListResponse,
    ClimateAlertResponse,
    TrainingResponse,
    LagdoDamResponse,
    FloodForecastDay,
    FloodForecastResponse,
)
from ..models.xgboost_flood import FloodXGBoost
from ..models.ensemble import FloodEnsemble
from ..models.compound_risk import batch_compound_risk
from ..pipeline.features import FEATURE_NAMES, merge_features, build_dynamic_features
from ..data.nimet import load_nimet_features
from ..data.openmeteo import fetch_karu_live_features, fetch_karu_forecast_series, fetch_karu_seasonal_series

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["climate"])

MODEL_DIR = os.getenv("MODEL_DIR", "models_store")

# NIMET data dir is populated by admin CSV uploads (volume-mounted in Docker)
_NIMET_DIR = os.getenv("NIMET_DATA_DIR", "nimet_data")
NIMET_RAINFALL_CSV = os.getenv("NIMET_RAINFALL_CSV", os.path.join(_NIMET_DIR, "dataset-rainfall.csv"))
NIMET_ET_CSV = os.getenv("NIMET_ET_CSV", os.path.join(_NIMET_DIR, "ET_dataset.csv"))

# The only LGA with real NIMET data — all predictions are for Karu only
KARU_LGA_ID = "nasarawa_karu"

# Zone-relative rainfall scaling factors (relative to middle-belt Karu)
_ZONE_FACTORS = {"south": 1.8, "middle": 1.0, "north": 0.45}

# Lazy-loaded singletons (populated on first request)
_model: FloodXGBoost | None = None
_static_features: pd.DataFrame | None = None
_lga_metadata: pd.DataFrame | None = None

# In-memory prediction cache — 5 min TTL matches Next.js revalidate: 300
_cache: list[dict] | None = None
_cache_ts: float = 0.0
_CACHE_TTL = 300


def _load_artifacts() -> None:
    """Load model and parquet files once, raising if missing."""
    global _model, _static_features, _lga_metadata

    if _model is not None:
        return

    model_path = os.path.join(MODEL_DIR, "flood_xgb.json")
    static_path = os.path.join(MODEL_DIR, "static_features.parquet")
    meta_path = os.path.join(MODEL_DIR, "lga_metadata.parquet")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Run scripts/train_local.py first."
        )

    _model = FloodXGBoost()
    _model.load(model_path, FEATURE_NAMES)

    if os.path.exists(static_path):
        _static_features = pd.read_parquet(static_path)
    else:
        raise FileNotFoundError(f"Static features not found at {static_path}.")

    if os.path.exists(meta_path):
        _lga_metadata = pd.read_parquet(meta_path)
    else:
        raise FileNotFoundError(f"LGA metadata not found at {meta_path}.")


def _get_karu_features() -> dict:
    """Return today's dynamic features for Karu LGA.

    Priority: Open-Meteo live data → NIMET CSV fallback → hardcoded defaults.
    """
    # 1. Try Open-Meteo — extract today's row from the same series used by the forecast
    #    so /flood-risk and /flood-forecast always agree on today's value.
    series = fetch_karu_forecast_series(forecast_days=16)
    if series is not None:
        today_rows = series[series["date"].dt.date == date.today()]
        row = today_rows.iloc[-1] if not today_rows.empty else series.iloc[-1]
        wb30 = float(row["water_balance_30d"])
        live = {
            "rain_1d":           float(row["rain_1d"]),
            "rain_3d":           float(row["rain_3d"]),
            "rain_7d":           float(row["rain_7d"]),
            "rain_14d":          float(row["rain_14d"]),
            "rain_30d":          float(row["rain_30d"]),
            "soil_moisture":     round(1.0 / (1.0 + math.exp(-wb30 / 50.0)), 4),
            "water_balance_7d":  float(row["water_balance_7d"]),
            "water_balance_30d": wb30,
            "data_date":         str(row["date"].date()),
        }
        logger.info("Karu features from Open-Meteo forecast series (data_date=%s)", live["data_date"])
        return live

    # 2. Fall back to NIMET CSVs if provided
    if NIMET_RAINFALL_CSV and os.path.exists(NIMET_RAINFALL_CSV):
        logger.warning("Open-Meteo unavailable — falling back to NIMET CSV")
        et_path = NIMET_ET_CSV if NIMET_ET_CSV and os.path.exists(NIMET_ET_CSV) else None
        return load_nimet_features(NIMET_RAINFALL_CSV, et_path)

    # 3. Last resort: mid-dry-season defaults
    logger.warning("All data sources unavailable — using hardcoded defaults")
    return {
        "rain_1d": 0.0,
        "rain_3d": 0.0,
        "rain_7d": 0.0,
        "rain_14d": 2.0,
        "rain_30d": 8.0,
        "soil_moisture": 0.25,
        "water_balance_7d": -12.0,
        "water_balance_30d": -35.0,
    }


def _build_dynamic(lga_metadata: pd.DataFrame, karu: dict) -> pd.DataFrame:
    """Scale Karu's dynamic features to all LGAs by geographic zone."""
    chirps: dict[str, dict[str, float]] = {}
    era5_soil: dict[str, float] = {}
    wb_map: dict[str, tuple[float, float]] = {}

    rain_keys = ["rain_1d", "rain_3d", "rain_7d", "rain_14d", "rain_30d"]

    for _, lga in lga_metadata.iterrows():
        lga_id = str(lga["lga_id"])
        is_karu = str(lga.get("name", "")).lower() == "karu"
        factor = 1.0 if is_karu else _ZONE_FACTORS.get(str(lga.get("zone", "middle")), 1.0)

        chirps[lga_id] = {k: karu[k] * factor for k in rain_keys}

        sm = karu.get("soil_moisture", 0.3)
        era5_soil[lga_id] = float(np.clip(sm + (sm - 0.3) * (factor - 1.0), 0.05, 0.95))

        wb_map[lga_id] = (
            karu.get("water_balance_7d", 0.0) * factor,
            karu.get("water_balance_30d", 0.0) * factor,
        )

    dynamic = build_dynamic_features(chirps, era5_soil, None, None)

    # Inject water balance columns (not produced by build_dynamic_features)
    wb7 = dynamic["lga_id"].map(lambda lid: wb_map.get(str(lid), (0.0, 0.0))[0])
    wb30 = dynamic["lga_id"].map(lambda lid: wb_map.get(str(lid), (0.0, 0.0))[1])
    dynamic["water_balance_7d"] = wb7.values
    dynamic["water_balance_30d"] = wb30.values

    return dynamic


def _run_prediction() -> list[dict]:
    """Predict flood risk for Karu LGA using live Open-Meteo data."""
    _load_artifacts()

    karu_raw = _get_karu_features()
    dynamic = _build_dynamic(_lga_metadata, karu_raw)

    features = merge_features(_static_features, dynamic)
    for col in FEATURE_NAMES:
        if col not in features.columns:
            features[col] = 0.0

    # Run model over all LGAs (needed to keep relative scores calibrated),
    # but only return Karu's prediction to the API.
    X = features[FEATURE_NAMES].to_numpy(dtype=float)
    xgb_preds = _model.predict(X)

    ensemble = FloodEnsemble()
    flood_probs = ensemble.predict(xgb_preds, None, np.zeros(len(X)))
    risk_levels = ensemble.classify_risk(flood_probs)

    karu_mask = features["lga_id"].astype(str) == KARU_LGA_ID
    karu_indices = [i for i, m in enumerate(karu_mask) if m]

    if not karu_indices:
        logger.error("Karu LGA (%s) not found in feature matrix", KARU_LGA_ID)
        return []

    compound_input = [
        {
            "lga_id": str(features.iloc[i]["lga_id"]),
            "flood_risk": float(flood_probs[i]),
            "health_vulnerability": 0.3,
            "exposure": 0.5,
            "adaptive_capacity": 0.5,
        }
        for i in karu_indices
    ]
    compound_results = batch_compound_risk(compound_input)

    today = karu_raw.get("data_date", str(date.today()))
    return [
        {
            "lga_id": compound["lga_id"],
            "prediction_date": today,
            "flood_probability": float(flood_probs[karu_indices[i]]),
            "risk_level": risk_levels[karu_indices[i]],
            "compound_score": compound["compound_score"],
            "compound_risk_level": compound["risk_level"],
        }
        for i, compound in enumerate(compound_results)
    ]


def _get_cached_predictions() -> list[dict]:
    global _cache, _cache_ts
    if _cache is not None and (time.time() - _cache_ts) < _CACHE_TTL:
        return _cache
    _cache = _run_prediction()
    _cache_ts = time.time()
    return _cache


# Separate cache for the 16-day forecast (heavier computation)
_forecast_cache: list[dict] | None = None
_forecast_cache_ts: float = 0.0


def _confidence(days_ahead: int) -> str:
    if days_ahead <= 2:
        return "high"
    if days_ahead <= 5:
        return "moderate"
    if days_ahead <= 16:
        return "indicative"
    return "seasonal"


def _day_label(d: "date", today: "date") -> str:
    delta = (d - today).days
    if delta == 0:
        return "Today"
    if delta == 1:
        return "Tomorrow"
    return d.strftime("%a %-d %b")


def _run_forecast() -> list[dict]:
    """Build 90-day flood forecast for Karu using Open-Meteo + XGBoost.

    Days 1-16: deterministic Open-Meteo forecast (high/moderate/indicative confidence).
    Days 17-90: Open-Meteo seasonal ensemble — median flood probability with
                p10/p90 rainfall uncertainty bands (confidence = "seasonal").
    """
    _load_artifacts()

    today = date.today()

    # --- 16-day deterministic series (30d history + 16d forecast) ---
    series_16 = fetch_karu_forecast_series(forecast_days=16)
    if series_16 is None:
        logger.warning("Open-Meteo forecast unavailable")
        return []

    # --- 92-day seasonal ensemble ---
    seasonal = fetch_karu_seasonal_series(forecast_days=92)

    # Build full combined series for rolling window continuity.
    # Base = 30d history + 16d forecast from deterministic API.
    # Extend with seasonal medians for days 17-92, estimating ET from
    # the mean ET observed in the 16-day forecast window.
    forecast_et_rows = series_16[series_16["is_forecast"]]
    mean_et = float(forecast_et_rows["et_mm"].mean()) if not forecast_et_rows.empty else 4.0

    if seasonal is not None:
        cutoff = today + timedelta(days=16)
        seasonal_ext = seasonal[seasonal["date"].dt.date > cutoff].copy()
        seasonal_ext = seasonal_ext.assign(et_mm=mean_et)
        # Merge into a combined frame for rolling window recomputation
        base = series_16[["date", "rain_mm", "et_mm"]].copy()
        combined = pd.concat(
            [base, seasonal_ext[["date", "rain_mm", "et_mm"]]],
            ignore_index=True,
        ).sort_values("date").reset_index(drop=True)
    else:
        combined = series_16[["date", "rain_mm", "et_mm"]].copy()
        seasonal_ext = pd.DataFrame()

    # Recompute rolling windows on the full combined series
    for w in [1, 3, 7, 14, 30]:
        combined[f"rain_{w}d"] = combined["rain_mm"].rolling(w, min_periods=1).sum()
    for w in [7, 30]:
        combined[f"et_{w}d"] = combined["et_mm"].rolling(w, min_periods=1).sum()
    combined["water_balance_7d"]  = combined["rain_7d"]  - combined["et_7d"]
    combined["water_balance_30d"] = combined["rain_30d"] - combined["et_30d"]
    combined["soil_moisture"] = combined["water_balance_30d"].apply(
        lambda wb: round(1.0 / (1.0 + math.exp(-wb / 50.0)), 4)
    )

    # Lookup table for p10/p90 from seasonal (keyed by date string)
    seasonal_bounds: dict[str, tuple[float, float]] = {}
    if seasonal is not None and not seasonal_ext.empty:
        for _, sr in seasonal_ext.iterrows():
            d = str(sr["date"].date())
            p10 = float(seasonal.loc[seasonal["date"] == sr["date"], "rain_mm_p10"].iloc[0])
            p90 = float(seasonal.loc[seasonal["date"] == sr["date"], "rain_mm_p90"].iloc[0])
            seasonal_bounds[d] = (round(p10, 1), round(p90, 1))

    # Slice to today onwards
    forecast_rows = combined[combined["date"].dt.date >= today].copy()
    if forecast_rows.empty:
        return []

    karu_static = _static_features[
        _static_features["lga_id"].astype(str) == KARU_LGA_ID
    ]
    if karu_static.empty:
        return []

    static_row = karu_static.iloc[0]
    ensemble = FloodEnsemble()
    import xgboost as xgb  # noqa: F401 — imported for model.predict

    results = []
    for _, row in forecast_rows.iterrows():
        feat = {col: 0.0 for col in FEATURE_NAMES}

        for col in ["hand_mean", "hand_min", "hand_std", "twi_mean", "twi_max",
                    "slope_mean", "population_density", "distance_to_river",
                    "flood_fraction_baseline", "land_cover_urban"]:
            feat[col] = float(static_row.get(col, 0))

        rain_30d = float(row["rain_30d"]) or 1.0
        feat["rain_1d"]  = float(row["rain_1d"])
        feat["rain_3d"]  = float(row["rain_3d"])
        feat["rain_7d"]  = float(row["rain_7d"])
        feat["rain_14d"] = float(row["rain_14d"])
        feat["rain_30d"] = float(row["rain_30d"])
        feat["rain_7d_ratio"]       = feat["rain_7d"] / rain_30d
        feat["rain_3d_ratio"]       = feat["rain_3d"] / rain_30d
        feat["soil_moisture"]       = float(row["soil_moisture"])
        feat["water_balance_7d"]    = float(row["water_balance_7d"])
        feat["water_balance_30d"]   = float(row["water_balance_30d"])
        feat["forecast_risk"]       = 0.0
        feat["benue_discharge_max"] = 0.0
        feat["niger_discharge_max"] = 0.0

        X = np.array([[feat[c] for c in FEATURE_NAMES]])
        xgb_pred = _model.predict(X)
        flood_prob = float(ensemble.predict(xgb_pred, None, np.zeros(1))[0])
        risk_level = ensemble.classify_risk(np.array([flood_prob]))[0]

        forecast_date = row["date"].date()
        days_ahead = (forecast_date - today).days
        date_str = str(forecast_date)
        bounds = seasonal_bounds.get(date_str)

        results.append({
            "date": date_str,
            "day_label": _day_label(forecast_date, today),
            "flood_probability": round(flood_prob, 4),
            "risk_level": risk_level,
            "rain_mm": round(float(row["rain_mm"]), 1),
            "rain_mm_p10": bounds[0] if bounds else None,
            "rain_mm_p90": bounds[1] if bounds else None,
            "confidence": _confidence(days_ahead),
            "is_forecast": days_ahead >= 0,
        })

    return results


_FORECAST_DISK_PATH = Path(os.environ.get("MODEL_STORE_PATH", "/app/models_store")) / "forecast_cache.json"


def _save_forecast_to_disk(forecasts: list[dict]) -> None:
    try:
        _FORECAST_DISK_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_FORECAST_DISK_PATH, "w") as f:
            json.dump({"saved_at": date.today().isoformat(), "forecasts": forecasts}, f)
    except Exception:
        pass


def _load_forecast_from_disk() -> list[dict]:
    try:
        with open(_FORECAST_DISK_PATH) as f:
            payload = json.load(f)
        forecasts = payload.get("forecasts", [])
        if forecasts:
            logger.warning("Serving stale forecast from disk (live fetch unavailable)")
        return forecasts
    except Exception:
        return []


def _get_cached_forecast() -> list[dict]:
    global _forecast_cache, _forecast_cache_ts
    if _forecast_cache is not None and (time.time() - _forecast_cache_ts) < _CACHE_TTL:
        return _forecast_cache
    fresh = _run_forecast()
    if fresh:
        _forecast_cache = fresh
        _forecast_cache_ts = time.time()
        _save_forecast_to_disk(fresh)
    else:
        # Live fetch failed — serve last known good data from disk
        _forecast_cache = _load_forecast_from_disk()
        _forecast_cache_ts = time.time()
    return _forecast_cache


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/flood-risk", response_model=FloodRiskListResponse)
async def get_flood_risk(
    target_date: date | None = None,
    lga_id: str | None = None,
    risk_level: str | None = None,
):
    """Get flood risk predictions for all LGAs (model-served, no DB)."""
    try:
        predictions = _get_cached_predictions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction pipeline failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    if lga_id:
        predictions = [p for p in predictions if p["lga_id"] == lga_id]
    if risk_level:
        predictions = [p for p in predictions if p["risk_level"] == risk_level]

    target = target_date or date.today()
    results = [
        FloodRiskResponse(
            lga_id=p["lga_id"],
            prediction_date=target,
            flood_probability=p["flood_probability"],
            risk_level=p["risk_level"],
            compound_score=p["compound_score"],
            compound_risk_level=p["compound_risk_level"],
        )
        for p in predictions
    ]

    return FloodRiskListResponse(predictions=results, date=target, total=len(results))


@router.post("/compound-risk", response_model=CompoundRiskListResponse)
async def get_compound_risk(request: CompoundRiskRequest):
    """Get compound vulnerability scores (flood + health proxy)."""
    try:
        predictions = _get_cached_predictions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Compound risk pipeline failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc

    if request.lga_ids:
        lga_set = set(request.lga_ids)
        predictions = [p for p in predictions if p["lga_id"] in lga_set]
    else:
        predictions = predictions[:100]

    results = [
        CompoundRiskResponse(
            lga_id=p["lga_id"],
            compound_score=p["compound_score"] or 0.0,
            risk_level=p["compound_risk_level"] or "unknown",
            components={"flood_probability": p["flood_probability"]},
        )
        for p in predictions
    ]

    return CompoundRiskListResponse(results=results, total=len(results))


@router.get("/alerts", response_model=ClimateAlertListResponse)
async def get_climate_alerts(
    limit: int = Query(50, le=200),
    risk_level: str | None = None,
):
    """Return alerts derived from current high-risk predictions."""
    try:
        predictions = _get_cached_predictions()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    high_risk = [
        p for p in predictions
        if p["risk_level"] in ("warning", "emergency")
        and (risk_level is None or p["risk_level"] == risk_level)
    ][:limit]

    alerts = [
        ClimateAlertResponse(
            id=f"alert_{p['lga_id']}_{date.today().isoformat()}",
            type="climate",
            risk_level=p["risk_level"],
            message=(
                f"Flood {p['risk_level']} for LGA {p['lga_id']}: "
                f"probability {p['flood_probability']:.1%}"
            ),
            lga_id=p["lga_id"],
            created_at=date.today().isoformat(),
        )
        for p in high_risk
    ]

    return ClimateAlertListResponse(alerts=alerts, total=len(alerts))


@router.get("/flood-forecast", response_model=FloodForecastResponse)
async def get_flood_forecast():
    """90-day flood outlook for Karu LGA powered by Open-Meteo + XGBoost.

    Days 1-16: deterministic forecast (high/moderate/indicative confidence).
    Days 17-90: seasonal ensemble median with p10/p90 uncertainty bands (confidence=seasonal).
    """
    try:
        _load_artifacts()
        forecasts = _get_cached_forecast()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Forecast pipeline failed")
        raise HTTPException(status_code=500, detail="Forecast failed") from exc

    if not forecasts:
        raise HTTPException(status_code=503, detail="Forecast data unavailable")

    return FloodForecastResponse(
        location="Karu LGA, Nasarawa",
        forecasts=[FloodForecastDay(**f) for f in forecasts],
        generated_at=date.today().isoformat(),
    )


@router.post("/retrain", response_model=TrainingResponse)
async def trigger_retrain(
    rainfall_file: UploadFile | None = File(None),
    et_file: UploadFile | None = File(None),
):
    """Upload new NIMET CSVs and retrain the XGBoost flood model.

    Accepts multipart uploads for rainfall and ET CSVs. Saves them to the
    NIMET data directory, runs the training script, then reloads the model.
    Can be called with just one file to update only that dataset.
    """
    import asyncio

    os.makedirs(_NIMET_DIR, exist_ok=True)

    saved = []

    if rainfall_file and rainfall_file.filename:
        dest = os.path.join(_NIMET_DIR, "dataset-rainfall.csv")
        content = await rainfall_file.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append("rainfall")
        logger.info("Saved new rainfall CSV (%d bytes) to %s", len(content), dest)

    if et_file and et_file.filename:
        dest = os.path.join(_NIMET_DIR, "ET_dataset.csv")
        content = await et_file.read()
        with open(dest, "wb") as f:
            f.write(content)
        saved.append("ET")
        logger.info("Saved new ET CSV (%d bytes) to %s", len(content), dest)

    rainfall_path = os.path.join(_NIMET_DIR, "dataset-rainfall.csv")
    et_path = os.path.join(_NIMET_DIR, "ET_dataset.csv")

    if not os.path.exists(rainfall_path):
        return TrainingResponse(
            status="error",
            message="Rainfall CSV not found. Upload a rainfall file first.",
        )

    # Run training in a subprocess so it doesn't block the event loop
    cmd = [
        "python", "scripts/train_local.py",
        "--nimet-csv", rainfall_path,
        "--output-dir", MODEL_DIR,
    ]
    if os.path.exists(et_path):
        cmd += ["--et-csv", et_path]

    logger.info("Starting retraining: %s", " ".join(cmd))

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd="/app" if os.path.exists("/app") else ".",
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=600)
    except asyncio.TimeoutError:
        return TrainingResponse(status="error", message="Training timed out after 10 minutes.")
    except Exception as exc:
        return TrainingResponse(status="error", message=f"Training failed: {exc}")

    if proc.returncode != 0:
        err = stderr.decode()[-500:] if stderr else "unknown error"
        logger.error("Retraining failed: %s", err)
        return TrainingResponse(status="error", message=f"Training script failed: {err}")

    # Reload model artifacts and clear caches
    global _cache, _cache_ts, _forecast_cache, _forecast_cache_ts, _model, _static_features, _lga_metadata
    _cache = None
    _cache_ts = 0.0
    _forecast_cache = None
    _forecast_cache_ts = 0.0
    _model = None
    _static_features = None
    _lga_metadata = None

    # Read metrics from the newly written file
    metrics_path = os.path.join(MODEL_DIR, "training_metrics.json")
    metrics = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

    files_msg = f" (updated: {', '.join(saved)})" if saved else ""
    return TrainingResponse(
        status="ok",
        message=f"Model retrained successfully{files_msg}. CV AUC: {metrics.get('cv_auc_mean', 'n/a')}",
        metrics=metrics,
    )


@router.get("/lagdo-dam", response_model=LagdoDamResponse)
async def get_lagdo_dam_status():
    """Get current Lagdo Dam risk status from GloFAS."""
    from ..data.glofas import (
        fetch_glofas_forecast,
        load_glofas_discharge,
        extract_station_discharge,
        check_lagdo_dam_risk,
        BENUE_STATIONS,
    )

    glofas_path = fetch_glofas_forecast(date.today())
    if not glofas_path:
        raise HTTPException(status_code=503, detail="GloFAS data unavailable")

    glofas_ds = load_glofas_discharge(glofas_path)
    benue_discharge = extract_station_discharge(glofas_ds, BENUE_STATIONS)
    risk = check_lagdo_dam_risk(benue_discharge)

    affected = [a.strip() for a in os.getenv("LAGDO_AFFECTED_LGAS", "").split(",") if a.strip()]

    return LagdoDamResponse(
        risk_level=risk.get("risk_level", "unknown"),
        discharge_m3s=risk.get("yola_discharge_m3s", 0),
        threshold_m3s=3000,
        affected_lgas=affected,
    )
