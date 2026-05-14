"""
backend/main.py  ── v5: Added /detect_pest endpoint for pest-detection screen
────────────────────────────────────────────────────────────────────────────────
Changes in v5:
  • Added POST /detect_pest endpoint — called by pest-detection.tsx with
    vibration signal data. Returns top-N pest matches with confidence,
    damage description, and treatment advice.
  • All existing v4 routes unchanged (/predict, /model-status, /sync-and-retrain,
    /health, /retrain-manual, /feature-importance, /firebase-preview)

Firebase structure (soil-app-483fa-default-rtdb):
  /soil_monitoring_system/History/{push_id}:
      moisture:       0           ← soil moisture (%)
      pest_status:    "Possible pest activity"  ← text → converted to 0/1
      temperature:    22.5        ← soil temperature (°C)
      timestamp:      "Thu Jan..."← ignored

Setup:
    set FIREBASE_URL=https://soil-app-483fa-default-rtdb.firebaseio.com
    set FIREBASE_HISTORY_PATH=soil_monitoring_system/History
    set FIREBASE_AUTH_TOKEN=your-token-if-db-is-private
    uvicorn main:app --reload --port 8000
────────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import List, Literal, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ─── Configuration ────────────────────────────────────────────────────────────

FIREBASE_URL           = os.getenv("FIREBASE_URL",           "https://soil-app-483fa-default-rtdb.firebaseio.com")
FIREBASE_AUTH_TOKEN    = os.getenv("FIREBASE_AUTH_TOKEN",    "")
FIREBASE_HISTORY_PATH  = os.getenv("FIREBASE_HISTORY_PATH",  "soil_monitoring_system/History")
RETRAIN_INTERVAL_HOURS = float(os.getenv("RETRAIN_INTERVAL_HOURS", "6"))
MIN_HISTORY_FOR_BLEND  = int(os.getenv("MIN_HISTORY_SAMPLES",      "20"))
MODEL_DIR              = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Feature / target definitions ─────────────────────────────────────────────

FEATURES = [
    "soil_moisture",
    "soil_temperature",
    "pest_presence",
    "air_temperature",
    "rainfall",
    "humidity",
    "wind_speed",
]

TARGETS = ["irrigation_needed", "pest_risk", "planting_window"]

# Lesotho / Maseru climatological means — imputed when weather fields are absent
LESOTHO_MEANS = {
    "air_temperature": 18.5,
    "rainfall":         1.8,
    "humidity":        62.0,
    "wind_speed":      10.0,
}

FIELD_ALIASES = {
    "moisture":          "soil_moisture",
    "temperature":       "soil_temperature",
    "pest_status":       "pest_presence",
    "airTemperature":    "air_temperature",
    "air_temperature":   "air_temperature",
    "rainfall":          "rainfall",
    "humidity":          "humidity",
    "windSpeed":         "wind_speed",
    "wind_speed":        "wind_speed",
    "soilMoisture":      "soil_moisture",
    "soil_moisture":     "soil_moisture",
    "soilTemperature":   "soil_temperature",
    "soil_temperature":  "soil_temperature",
    "pestPresence":      "pest_presence",
    "pest_presence":     "pest_presence",
}

NO_PEST_PHRASES = {"no pest", "clear", "none", "normal", "no pest detected", "no pest activity"}

PEST_MAP     = {0: "low",   1: "medium",     2: "high"}
PLANTING_MAP = {0: "avoid", 1: "suboptimal", 2: "optimal"}

# ─── Pest species database ────────────────────────────────────────────────────
# Used by /detect_pest to match vibration signatures.
# Each entry defines the frequency band (Hz) where the pest is most active,
# plus agronomic info for the app UI.
#
PEST_PROFILES = [
    {
        "pest":       "African Armyworm (Spodoptera exempta)",
        "freq_lo":    80,
        "freq_hi":    250,
        "damage":     "Destroys maize and sorghum leaves overnight. Can strip an entire field in days.",
        "treatment":  "Apply lambda-cyhalothrin or emamectin benzoate at dusk when larvae are active.",
    },
    {
        "pest":       "Termites (Macrotermes spp.)",
        "freq_lo":    20,
        "freq_hi":    100,
        "damage":     "Attacks roots and stems of maize, sorghum, and wheat. Causes sudden wilting.",
        "treatment":  "Apply chlorpyrifos drench around stem base. Remove crop residues after harvest.",
    },
    {
        "pest":       "Mole Cricket (Gryllotalpa africana)",
        "freq_lo":    150,
        "freq_hi":    400,
        "damage":     "Tunnels through soil, severing roots and seedlings. Most damaging in sandy soils.",
        "treatment":  "Bait with carbaryl-treated bran at night. Flood tunnels with soapy water.",
    },
    {
        "pest":       "White Grub / Scarab Beetle Larva",
        "freq_lo":    50,
        "freq_hi":    180,
        "damage":     "Feeds on roots of maize, potatoes, and pasture grasses. Causes patchy die-off.",
        "treatment":  "Apply imidacloprid granules at planting. Rotate crops annually.",
    },
    {
        "pest":       "Cutworm (Agrotis spp.)",
        "freq_lo":    100,
        "freq_hi":    300,
        "damage":     "Cuts seedlings at soil level. Most active in cool, moist soils at night.",
        "treatment":  "Apply chlorpyrifos or spinosad bait in furrows. Cultivate to expose pupae.",
    },
    {
        "pest":       "Root-Knot Nematode (Meloidogyne spp.)",
        "freq_lo":    10,
        "freq_hi":    60,
        "damage":     "Causes galls on roots of vegetables and legumes, reducing water and nutrient uptake.",
        "treatment":  "Solarise soil, apply carbofuran, or rotate with resistant varieties.",
    },
    {
        "pest":       "Wireworm (Agriotes spp.)",
        "freq_lo":    40,
        "freq_hi":    130,
        "damage":     "Bores into potato tubers, maize kernels, and cereal seeds. Hard to detect early.",
        "treatment":  "Seed treatment with thiamethoxam. Avoid planting in fields with grass history.",
    },
]


def _dominant_frequency(signal: list, sampling_rate: int) -> float:
    """
    Returns the dominant frequency (Hz) in a vibration signal using FFT.
    Used to match against pest frequency profiles.
    """
    arr  = np.array(signal, dtype=float)
    n    = len(arr)
    if n < 2:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(arr - arr.mean()))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    # Ignore DC component (index 0)
    if len(fft_vals) < 2:
        return 0.0
    dominant_idx = np.argmax(fft_vals[1:]) + 1
    return float(freqs[dominant_idx])


def _match_pests(signal: list, sampling_rate: int, top_n: int = 3) -> list:
    """
    Scores each pest profile against the dominant frequency of the signal.
    Returns top_n matches sorted by score descending.
    """
    dom_freq  = _dominant_frequency(signal, sampling_rate)
    amplitude = float(np.std(signal))  # proxy for activity intensity

    results = []
    for profile in PEST_PROFILES:
        lo, hi = profile["freq_lo"], profile["freq_hi"]
        center = (lo + hi) / 2.0
        width  = (hi - lo) / 2.0

        # Gaussian-like score: 1.0 if dominant freq is at profile centre, falls off outside band
        if width > 0:
            freq_score = max(0.0, 1.0 - abs(dom_freq - center) / (width * 2))
        else:
            freq_score = 1.0 if lo <= dom_freq <= hi else 0.0

        # Amplitude bonus: higher vibration = more confident detection
        amplitude_bonus = min(0.2, amplitude * 0.1)
        raw_score = freq_score + amplitude_bonus

        # Normalise to 0–100
        score = round(min(100, raw_score * 100), 1)

        if score >= 5.0:  # filter out near-zero matches
            confidence = "High" if score >= 65 else "Medium" if score >= 35 else "Low"
            results.append({
                "pest":       profile["pest"],
                "score":      score,
                "confidence": confidence,
                "damage":     profile["damage"],
                "treatment":  profile["treatment"],
            })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


# ─── Global model state ───────────────────────────────────────────────────────

_state: dict = {
    "models": {},
    "scaler": None,
    "meta": {
        "last_trained":       None,
        "seed_samples":       0,
        "history_samples":    0,
        "total_samples":      0,
        "is_training":        False,
        "firebase_connected": False,
        "history_path":       FIREBASE_HISTORY_PATH,
        "skipped_rows":       0,
        "accuracy": {
            "irrigation_needed": None,
            "pest_risk":         None,
            "planting_window":   None,
        },
    },
}

_retrain_lock = threading.Lock()

# ─── Firebase REST helper ─────────────────────────────────────────────────────

def firebase_get(path: str) -> Optional[dict]:
    url    = f"{FIREBASE_URL}/{path}.json"
    params = {"auth": FIREBASE_AUTH_TOKEN} if FIREBASE_AUTH_TOKEN else {}
    try:
        resp = requests.get(url, params=params, timeout=20)
        if resp.status_code == 200:
            return resp.json()
        print(f"[Firebase] GET /{path} → HTTP {resp.status_code}: {resp.text[:300]}")
        return None
    except Exception as exc:
        print(f"[Firebase] Connection error: {exc}")
        return None

# ─── pest_status text → 0 / 1 ────────────────────────────────────────────────

def _parse_pest_status(value) -> int:
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in NO_PEST_PHRASES:
            return 0
        if "no pest" in cleaned or cleaned == "clear":
            return 0
        return 1
    return 0

# ─── Firebase history → labelled DataFrame ────────────────────────────────────

def fetch_history_df() -> pd.DataFrame:
    raw = firebase_get(FIREBASE_HISTORY_PATH)
    if not raw or not isinstance(raw, dict):
        print("[Firebase] No history data found.")
        return pd.DataFrame()

    rows      = []
    skipped   = 0
    total_raw = 0

    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        total_raw += 1
        row: dict = {}

        for fb_field, our_field in FIELD_ALIASES.items():
            if fb_field in entry and our_field not in row:
                raw_val = entry[fb_field]
                if our_field == "pest_presence":
                    row[our_field] = _parse_pest_status(raw_val)
                else:
                    row[our_field] = raw_val

        rows.append(row)

    if not rows:
        print("[Firebase] No parseable rows found.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    for col, default in LESOTHO_MEANS.items():
        if col not in df.columns:
            df[col] = default
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    for col in FEATURES:
        if col not in df.columns:
            df[col] = np.nan
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["soil_moisture", "soil_temperature", "pest_presence"], inplace=True)
    df.dropna(subset=FEATURES, inplace=True)

    valid_mask = (
        (df["soil_temperature"] >= -10) &
        (df["soil_temperature"] <= 60)  &
        (df["soil_moisture"]    >=  0)  &
        (df["soil_moisture"]    <= 100)
    )
    df     = df[valid_mask].copy()
    skipped = total_raw - len(df)

    if df.empty:
        print(f"[Firebase] ⚠️  All {total_raw} rows were invalid.")
        return df

    df = _derive_labels(df)
    _state["meta"]["firebase_connected"] = True
    _state["meta"]["skipped_rows"]       = skipped

    print(f"[Firebase] ✅ Loaded {len(df)} usable rows "
          f"(skipped {skipped} invalid out of {total_raw} total).")
    return df

# ─── Label derivation ─────────────────────────────────────────────────────────

def _derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    sm  = df["soil_moisture"].values
    st  = df["soil_temperature"].values
    pp  = df["pest_presence"].values
    at  = df["air_temperature"].values
    rf  = df["rainfall"].values
    hum = df["humidity"].values

    df["irrigation_needed"] = ((sm < 35) & (rf < 2)).astype(int)

    pest_score = (
        pp * 0.50 +
        (hum > 70).astype(float) * 0.25 +
        (st  > 25).astype(float) * 0.15 +
        (at  > 28).astype(float) * 0.10
    )
    df["pest_risk"] = np.where(pest_score > 0.6, 2, np.where(pest_score > 0.3, 1, 0))

    planting_score = (
        ((sm >= 40) & (sm <= 70)).astype(float) * 0.35 +
        ((st >= 15) & (st <= 25)).astype(float) * 0.30 +
        (rf <  3).astype(float)                  * 0.20 +
        ((at >= 18) & (at <= 28)).astype(float) * 0.15
    )
    df["planting_window"] = np.where(
        planting_score > 0.70, 2,
        np.where(planting_score > 0.40, 1, 0)
    )
    return df

# ─── Seed data generator ──────────────────────────────────────────────────────

def generate_lesotho_seed_data(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df  = pd.DataFrame({
        "soil_moisture":    rng.uniform(5,  95, n),
        "soil_temperature": rng.uniform(5,  38, n),
        "pest_presence":    rng.choice([0, 1], n, p=[0.65, 0.35]),
        "air_temperature":  rng.uniform(2,  34, n),
        "rainfall":         rng.exponential(2.5, n).clip(0, 40),
        "humidity":         rng.uniform(25, 95, n),
        "wind_speed":       rng.uniform(0,  40, n),
    })
    return _derive_labels(df)

# ─── Model trainer ────────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame, history_count: int = 0):
    _state["meta"]["is_training"] = True
    print(f"\n🌲 Training on {len(df)} samples ({history_count} from Firebase)…")

    X        = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models     = {}
    accuracies = {}

    for target in TARGETS:
        y = df[target].values
        try:
            Xtr, Xte, ytr, yte = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr)
        acc                = float(accuracy_score(yte, clf.predict(Xte)))
        models[target]     = clf
        accuracies[target] = round(acc, 4)
        print(f"  [{target}] accuracy = {acc:.3f}")

    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    for name, model in models.items():
        joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")

    _state["models"] = models
    _state["scaler"] = scaler
    _state["meta"].update({
        "last_trained":    datetime.utcnow().isoformat() + "Z",
        "seed_samples":    len(df) - history_count,
        "history_samples": history_count,
        "total_samples":   len(df),
        "is_training":     False,
        "accuracy":        accuracies,
    })

    with open(f"{MODEL_DIR}/meta.json", "w") as fh:
        json.dump(_state["meta"], fh, indent=2)

    print(f"✅ Training complete — {len(df)} total, {history_count} from Firebase.\n")

# ─── Sync + train pipeline ────────────────────────────────────────────────────

def sync_and_train():
    with _retrain_lock:
        df_history    = fetch_history_df()
        history_count = len(df_history)

        seed_n  = max(200, 2000 - history_count * 3)
        df_seed = generate_lesotho_seed_data(seed_n)

        if history_count >= MIN_HISTORY_FOR_BLEND:
            df_combined = pd.concat([df_seed, df_history], ignore_index=True)
            print(f"[Blend] {seed_n} seed + {history_count} Firebase = {len(df_combined)} rows")
        else:
            df_combined = df_seed
            print(f"[Blend] Firebase history too small ({history_count} rows) — seed only for now")

        train_models(df_combined, history_count=history_count)

# ─── Background auto-retrain ──────────────────────────────────────────────────

def _auto_retrain_loop():
    while True:
        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)
        print(f"\n⏰ Scheduled retrain (every {RETRAIN_INTERVAL_HOURS}h)")
        sync_and_train()

# ─── Startup ──────────────────────────────────────────────────────────────────

def _startup():
    model_paths = [f"{MODEL_DIR}/scaler.pkl"] + [f"{MODEL_DIR}/{t}.pkl" for t in TARGETS]

    if all(os.path.exists(p) for p in model_paths):
        _state["scaler"] = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        _state["models"] = {t: joblib.load(f"{MODEL_DIR}/{t}.pkl") for t in TARGETS}
        meta_path = f"{MODEL_DIR}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                _state["meta"].update(json.load(fh))
        print("✅ Models loaded. Syncing Firebase history in background…")
        threading.Thread(target=sync_and_train, daemon=True).start()
    else:
        print("🌱 First run — training on seed + Firebase history…")
        sync_and_train()

    threading.Thread(target=_auto_retrain_loop, daemon=True).start()

_startup()

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="SoilApp AI Backend",
    description="Random Forest models that learn continuously from Firebase sensor history",
    version="5.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Schemas ──────────────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    soil_moisture:    float = Field(..., ge=0,   le=100)
    soil_temperature: float = Field(..., ge=-10, le=60)
    pest_presence:    int   = Field(..., ge=0,   le=1)
    air_temperature:  float = Field(..., ge=-10, le=50)
    rainfall:         float = Field(..., ge=0)
    humidity:         float = Field(..., ge=0,   le=100)
    wind_speed:       float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    irrigationNeeded:     bool
    irrigationConfidence: float
    pestRisk:             Literal["low", "medium", "high"]
    pestRiskScore:        float
    plantingWindow:       Literal["optimal", "suboptimal", "avoid"]
    plantingScore:        float
    recommendation:       str
    featureImportance:    dict
    modelMeta:            dict

class ManualReadings(BaseModel):
    readings: list[dict]

# NEW in v5 ───────────────────────────────────────────────────────────────────
class PestDetectInput(BaseModel):
    signal:           List[float] = Field(..., description="Vibration samples in m/s²")
    sampling_rate:    int         = Field(10000, ge=100, le=50000, description="Hz")
    sensor_depth_cm:  float       = Field(10.0,  ge=0,  le=100,   description="Depth of sensor in soil (cm)")
    top_n:            int         = Field(3,      ge=1,  le=7,     description="Number of top matches to return")

class PestMatch(BaseModel):
    pest:       str
    score:      float
    confidence: str
    damage:     str
    treatment:  str

class PestDetectResponse(BaseModel):
    dominant_frequency_hz: float
    signal_std:            float
    sensor_depth_cm:       float
    matches:               List[PestMatch]
    summary:               str
# ─────────────────────────────────────────────────────────────────────────────

# ─── Recommendation builder ───────────────────────────────────────────────────

def build_recommendation(irr: bool, pest: str, planting: str) -> str:
    if irr:
        return "⚠️ Soil moisture is critically low. Irrigate within 24 hours to prevent crop stress."
    if pest == "high":
        return "🐛 High pest risk detected. Apply preventive treatment and inspect crops closely."
    if pest == "medium":
        return "🔎 Moderate pest risk. Monitor field edges — conditions favour pest activity."
    if planting == "optimal":
        return "🌱 Excellent conditions for maize or sorghum. Soil and Lesotho weather are ideal — plant now."
    if planting == "suboptimal":
        return "📊 Conditions acceptable but not ideal. Wait 1–2 days for improvement."
    return "✅ All indicators are safe. No immediate action required. Monitor daily."

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":        "ok",
        "version":       "5.0.0",
        "models_loaded": list(_state["models"].keys()),
        "is_training":   _state["meta"]["is_training"],
    }


@app.get("/model-status")
def model_status():
    """Returns training metadata for the AI Learning Status panel in the app."""
    return _state["meta"]


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorInput):
    if not _state["models"] or _state["scaler"] is None:
        raise HTTPException(status_code=503, detail="Models are still initialising. Retry shortly.")

    X  = np.array([[
        data.soil_moisture, data.soil_temperature, data.pest_presence,
        data.air_temperature, data.rainfall, data.humidity, data.wind_speed,
    ]])
    Xs = _state["scaler"].transform(X)

    irr_p  = _state["models"]["irrigation_needed"].predict_proba(Xs)[0]
    irr_c  = int(_state["models"]["irrigation_needed"].predict(Xs)[0])

    pest_p = _state["models"]["pest_risk"].predict_proba(Xs)[0]
    pest_c = int(_state["models"]["pest_risk"].predict(Xs)[0])

    plan_p = _state["models"]["planting_window"].predict_proba(Xs)[0]
    plan_c = int(_state["models"]["planting_window"].predict(Xs)[0])

    pest_label     = PEST_MAP[pest_c]
    planting_label = PLANTING_MAP[plan_c]

    importance = dict(zip(
        FEATURES,
        _state["models"]["irrigation_needed"].feature_importances_.round(3).tolist()
    ))

    return PredictionResponse(
        irrigationNeeded=bool(irr_c),
        irrigationConfidence=float(irr_p[irr_c]),
        pestRisk=pest_label,
        pestRiskScore=float(pest_p[pest_c]),
        plantingWindow=planting_label,
        plantingScore=float(plan_p[plan_c]),
        recommendation=build_recommendation(bool(irr_c), pest_label, planting_label),
        featureImportance=importance,
        modelMeta=_state["meta"],
    )


@app.post("/sync-and-retrain")
def force_retrain(background_tasks: BackgroundTasks):
    """Force an immediate Firebase sync + retrain (runs in background)."""
    if _retrain_lock.locked():
        return {"status": "already_running", "message": "Training already in progress."}
    background_tasks.add_task(sync_and_train)
    return {"status": "started", "message": "Firebase sync and retrain started."}


@app.post("/retrain-manual")
def retrain_manual(body: ManualReadings, background_tasks: BackgroundTasks):
    """Supply additional readings (labels derived automatically). Min 10 required."""
    if len(body.readings) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 readings.")

    def _task():
        df_manual   = _derive_labels(pd.DataFrame(body.readings))
        df_history  = fetch_history_df()
        df_seed     = generate_lesotho_seed_data(500)
        frames      = [df_seed, df_manual]
        if not df_history.empty:
            frames.insert(1, df_history)
        df_combined   = pd.concat(frames, ignore_index=True)
        history_count = len(df_history) + len(df_manual)
        train_models(df_combined, history_count=history_count)

    background_tasks.add_task(_task)
    return {"status": "started", "submitted_readings": len(body.readings)}


@app.get("/feature-importance")
def feature_importance():
    """Per-model feature importance scores."""
    if not _state["models"]:
        raise HTTPException(status_code=503, detail="Models not yet ready.")
    return {
        name: dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
        for name, model in _state["models"].items()
    }


@app.get("/firebase-preview")
def firebase_preview():
    """Debug: shows first 3 raw Firebase records and how they are parsed."""
    raw = firebase_get(FIREBASE_HISTORY_PATH)
    if not raw or not isinstance(raw, dict):
        return {"error": "No data found at path", "path": FIREBASE_HISTORY_PATH}

    preview = []
    for i, (key, entry) in enumerate(raw.items()):
        if i >= 3:
            break
        if not isinstance(entry, dict):
            continue
        parsed = {}
        for fb_field, our_field in FIELD_ALIASES.items():
            if fb_field in entry and our_field not in parsed:
                raw_val = entry[fb_field]
                if our_field == "pest_presence":
                    parsed[our_field] = _parse_pest_status(raw_val)
                else:
                    parsed[our_field] = raw_val
        preview.append({"firebase_key": key, "raw": entry, "parsed": parsed})

    return {"path": FIREBASE_HISTORY_PATH, "total_records": len(raw), "preview": preview}


# ── NEW in v5 ─────────────────────────────────────────────────────────────────

@app.post("/detect_pest", response_model=PestDetectResponse)
def detect_pest(data: PestDetectInput):
    """
    Analyses a vibration signal from the soil sensor and identifies the most
    likely pest species based on dominant frequency matching.

    Called by pest-detection.tsx after the ESP32 uploads vibration chunks
    to Firebase and the app assembles the full signal.

    Request body:
        signal          – list of float (m/s²), min ~1000 samples recommended
        sampling_rate   – int Hz (default 10000, must match ESP32 SAMPLE_RATE)
        sensor_depth_cm – float, depth sensor was buried (affects freq weighting)
        top_n           – int, number of top matches to return (default 3)

    Response:
        dominant_frequency_hz – the peak frequency found in the signal
        signal_std            – standard deviation (proxy for vibration intensity)
        sensor_depth_cm       – echoed back from request
        matches               – list of PestMatch sorted by score descending
        summary               – human-readable one-liner for the app
    """
    if len(data.signal) < 100:
        raise HTTPException(
            status_code=400,
            detail=f"Signal too short ({len(data.signal)} samples). Need at least 100."
        )

    dom_freq   = _dominant_frequency(data.signal, data.sampling_rate)
    signal_std = float(np.std(data.signal))
    matches    = _match_pests(data.signal, data.sampling_rate, top_n=data.top_n)

    if not matches:
        summary = "No significant pest vibration signatures detected in this sample."
    elif matches[0]["confidence"] == "High":
        summary = f"Strong match: {matches[0]['pest']} detected. Immediate inspection recommended."
    elif matches[0]["confidence"] == "Medium":
        summary = f"Possible activity: {matches[0]['pest']}. Monitor closely over next 24–48 hours."
    else:
        summary = "Low-level vibration detected. Could be environmental noise or early-stage activity."

    return PestDetectResponse(
        dominant_frequency_hz=round(dom_freq, 2),
        signal_std=round(signal_std, 6),
        sensor_depth_cm=data.sensor_depth_cm,
        matches=[PestMatch(**m) for m in matches],
        summary=summary,
    )