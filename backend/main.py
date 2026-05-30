import json
import os
import random
import threading
import time
from datetime import datetime
from typing import List, Literal, Optional

import africastalking
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

AT_USERNAME = os.getenv("AT_USERNAME", "").strip()
AT_API_KEY  = os.getenv("AT_API_KEY",  "").strip()

os.makedirs(MODEL_DIR, exist_ok=True)

# ─── Africa's Talking initialisation ─────────────────────────────────────────

if not AT_USERNAME or not AT_API_KEY:
    print("WARNING: AT_USERNAME or AT_API_KEY is not set - SMS will fail.")
elif AT_USERNAME.lower() == "sandbox":
    print("WARNING: AT_USERNAME is 'sandbox' - SMS will only work in the AT simulator.")
else:
    print(f"Africa's Talking initialised in LIVE mode (username: {AT_USERNAME})")

africastalking.initialize(username=AT_USERNAME, api_key=AT_API_KEY)
sms_client = africastalking.SMS

last_sms_sent: dict = {}
SMS_COOLDOWN         = 3600

last_sms_hash_sent = ""

otp_store: dict = {}

# ─── Feature / target definitions ─────────────────────────────────────────────

FEATURES = [
    "soil_moisture", "soil_temperature", "pest_presence",
    "air_temperature", "rainfall", "humidity", "wind_speed",
]
TARGETS = ["irrigation_needed", "pest_risk", "planting_window"]

LESOTHO_MEANS = {
    "air_temperature": 18.5,
    "rainfall":         1.8,
    "humidity":        62.0,
    "wind_speed":      10.0,
}

FIELD_ALIASES = {
    "moisture":         "soil_moisture",
    "temperature":      "soil_temperature",
    "pest_status":      "pest_presence",
    "airTemperature":   "air_temperature",
    "air_temperature":  "air_temperature",
    "rainfall":         "rainfall",
    "humidity":         "humidity",
    "windSpeed":        "wind_speed",
    "wind_speed":       "wind_speed",
    "soilMoisture":     "soil_moisture",
    "soil_moisture":    "soil_moisture",
    "soilTemperature":  "soil_temperature",
    "soil_temperature": "soil_temperature",
    "pestPresence":     "pest_presence",
    "pest_presence":    "pest_presence",
}

NO_PEST_PHRASES = {"no pest", "clear", "none", "normal", "no pest detected", "no pest activity"}

PEST_MAP     = {0: "low",   1: "medium",     2: "high"}
PLANTING_MAP = {0: "avoid", 1: "suboptimal", 2: "optimal"}

PEST_PROFILES = [
    {"pest": "African Armyworm (Spodoptera exempta)", "freq_lo": 80,  "freq_hi": 250,
     "damage": "Destroys maize and sorghum leaves overnight.",
     "treatment": "Apply lambda-cyhalothrin or emamectin benzoate at dusk."},
    {"pest": "Termites (Macrotermes spp.)",           "freq_lo": 20,  "freq_hi": 100,
     "damage": "Attacks roots and stems of maize, sorghum, and wheat.",
     "treatment": "Apply chlorpyrifos drench around stem base."},
    {"pest": "Mole Cricket (Gryllotalpa africana)",   "freq_lo": 150, "freq_hi": 400,
     "damage": "Tunnels through soil, severing roots and seedlings.",
     "treatment": "Bait with carbaryl-treated bran at night."},
    {"pest": "White Grub / Scarab Beetle Larva",      "freq_lo": 50,  "freq_hi": 180,
     "damage": "Feeds on roots of maize, potatoes, and pasture grasses.",
     "treatment": "Apply imidacloprid granules at planting."},
    {"pest": "Cutworm (Agrotis spp.)",                "freq_lo": 100, "freq_hi": 300,
     "damage": "Cuts seedlings at soil level at night.",
     "treatment": "Apply chlorpyrifos or spinosad bait in furrows."},
    {"pest": "Root-Knot Nematode (Meloidogyne spp.)", "freq_lo": 10,  "freq_hi": 60,
     "damage": "Causes galls on roots of vegetables and legumes.",
     "treatment": "Solarise soil or rotate with resistant varieties."},
    {"pest": "Wireworm (Agriotes spp.)",              "freq_lo": 40,  "freq_hi": 130,
     "damage": "Bores into potato tubers, maize kernels, and cereal seeds.",
     "treatment": "Seed treatment with thiamethoxam."},
]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_phone(phone: str) -> str:
    phone = phone.strip().replace(" ", "").replace("-", "")
    if phone.startswith("00266"):
        return "+" + phone[2:]
    if phone.startswith("266") and not phone.startswith("+"):
        return "+" + phone
    if not phone.startswith("+"):
        return "+266" + phone
    return phone


def _dominant_frequency(signal: list, sampling_rate: int) -> float:
    arr = np.array(signal, dtype=float)
    n   = len(arr)
    if n < 2:
        return 0.0
    fft_vals = np.abs(np.fft.rfft(arr - arr.mean()))
    freqs    = np.fft.rfftfreq(n, d=1.0 / sampling_rate)
    if len(fft_vals) < 2:
        return 0.0
    dominant_idx = np.argmax(fft_vals[1:]) + 1
    return float(freqs[dominant_idx])


def _match_pests(signal: list, sampling_rate: int, top_n: int = 3) -> list:
    dom_freq  = _dominant_frequency(signal, sampling_rate)
    amplitude = float(np.std(signal))
    results   = []
    for profile in PEST_PROFILES:
        lo, hi   = profile["freq_lo"], profile["freq_hi"]
        center   = (lo + hi) / 2.0
        width    = (hi - lo) / 2.0
        freq_score = max(0.0, 1.0 - abs(dom_freq - center) / (width * 2)) if width > 0 else (1.0 if lo <= dom_freq <= hi else 0.0)
        amplitude_bonus = min(0.2, amplitude * 0.1)
        score = round(min(100, (freq_score + amplitude_bonus) * 100), 1)
        if score >= 5.0:
            confidence = "High" if score >= 65 else "Medium" if score >= 35 else "Low"
            results.append({"pest": profile["pest"], "score": score, "confidence": confidence,
                             "damage": profile["damage"], "treatment": profile["treatment"]})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]

# ─── Global model state ───────────────────────────────────────────────────────

_state: dict = {
    "models": {},
    "scaler": None,
    "meta": {
        "last_trained": None, "seed_samples": 0, "history_samples": 0,
        "total_samples": 0, "is_training": False, "firebase_connected": False,
        "history_path": FIREBASE_HISTORY_PATH, "skipped_rows": 0,
        "accuracy": {"irrigation_needed": None, "pest_risk": None, "planting_window": None},
    },
}

_retrain_lock = threading.Lock()

# ─── Firebase REST helpers ────────────────────────────────────────────────────

def firebase_get(path: str) -> Optional[dict]:
    url    = f"{FIREBASE_URL}/{path}.json"
    params = {"auth": FIREBASE_AUTH_TOKEN} if FIREBASE_AUTH_TOKEN else {}
    try:
        resp = requests.get(url, params=params, timeout=20)
        return resp.json() if resp.status_code == 200 else None
    except Exception as exc:
        print(f"[Firebase] Connection error: {exc}")
        return None


def firebase_set(path: str, data: dict) -> bool:
    """Overwrite a Firebase node (PUT)."""
    url    = f"{FIREBASE_URL}/{path}.json"
    params = {"auth": FIREBASE_AUTH_TOKEN} if FIREBASE_AUTH_TOKEN else {}
    try:
        resp = requests.put(url, json=data, params=params, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        print(f"[Firebase] Write error: {exc}")
        return False


def firebase_push(path: str, data: dict) -> bool:
    """
    Append a new child to a Firebase list (POST).
    Firebase auto-generates a unique push key.
    Used for sms_log so every sent SMS is kept — nothing is overwritten.
    """
    url    = f"{FIREBASE_URL}/{path}.json"
    params = {"auth": FIREBASE_AUTH_TOKEN} if FIREBASE_AUTH_TOKEN else {}
    try:
        resp = requests.post(url, json=data, params=params, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        print(f"[Firebase] Push error: {exc}")
        return False

# ─── pest_status text → 0/1 ───────────────────────────────────────────────────

def _parse_pest_status(value) -> int:
    if isinstance(value, (int, float)):
        return int(bool(value))
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in NO_PEST_PHRASES or "no pest" in cleaned or cleaned == "clear":
            return 0
        return 1
    return 0

# ─── Firebase history → labelled DataFrame ────────────────────────────────────

def fetch_history_df() -> pd.DataFrame:
    raw = firebase_get(FIREBASE_HISTORY_PATH)
    if not raw or not isinstance(raw, dict):
        return pd.DataFrame()

    rows = []
    total_raw = 0
    for entry in raw.values():
        if not isinstance(entry, dict):
            continue
        total_raw += 1
        row: dict = {}
        for fb_field, our_field in FIELD_ALIASES.items():
            if fb_field in entry and our_field not in row:
                raw_val = entry[fb_field]
                row[our_field] = _parse_pest_status(raw_val) if our_field == "pest_presence" else raw_val
        rows.append(row)

    if not rows:
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
        (df["soil_temperature"] >= -10) & (df["soil_temperature"] <= 60) &
        (df["soil_moisture"]    >=   0) & (df["soil_moisture"]    <= 100)
    )
    df      = df[valid_mask].copy()
    skipped = total_raw - len(df)
    if df.empty:
        return df
    df = _derive_labels(df)
    _state["meta"]["firebase_connected"] = True
    _state["meta"]["skipped_rows"]       = skipped
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
    pest_score = pp * 0.50 + (hum > 70).astype(float) * 0.25 + (st > 25).astype(float) * 0.15 + (at > 28).astype(float) * 0.10
    df["pest_risk"] = np.where(pest_score > 0.6, 2, np.where(pest_score > 0.3, 1, 0))
    planting_score = (
        ((sm >= 40) & (sm <= 70)).astype(float) * 0.35 +
        ((st >= 15) & (st <= 25)).astype(float) * 0.30 +
        (rf < 3).astype(float)                   * 0.20 +
        ((at >= 18) & (at <= 28)).astype(float) * 0.15
    )
    df["planting_window"] = np.where(planting_score > 0.70, 2, np.where(planting_score > 0.40, 1, 0))
    return df

# ─── Seed data ────────────────────────────────────────────────────────────────

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
    X        = df[FEATURES].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    models   = {}
    accuracies = {}
    for target in TARGETS:
        y = df[target].values
        try:
            Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc = float(accuracy_score(yte, clf.predict(Xte)))
        models[target]     = clf
        accuracies[target] = round(acc, 4)

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

# ─── Sync + train pipeline ────────────────────────────────────────────────────

def sync_and_train():
    with _retrain_lock:
        df_history    = fetch_history_df()
        history_count = len(df_history)
        seed_n        = max(200, 2000 - history_count * 3)
        df_seed       = generate_lesotho_seed_data(seed_n)
        if history_count >= MIN_HISTORY_FOR_BLEND:
            df_combined = pd.concat([df_seed, df_history], ignore_index=True)
        else:
            df_combined = df_seed
        train_models(df_combined, history_count=history_count)


def _auto_retrain_loop():
    while True:
        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)
        sync_and_train()

# ─── SMS: fetch farmers + send + SAVE TO FIREBASE ────────────────────────────

def auto_send_sms_to_farmers(sms_text: str):
    """
    1. Sends the SMS to every smsEnabled farmer in Firebase.
    2. After sending, saves a record to TWO Firebase paths:

       sms_log/{push_key}/          ← full history, one entry per send
         sentAt:          "2026-05-30T11:44:00Z"
         sentAtMs:        1748598240000
         message:         "the full SMS text exactly as sent"
         recipientCount:  2
         recipients:      [ { name, phone, status }, ... ]

       sms_log_latest/              ← always the most recent SMS (fixed path)
         (same fields as above)

    alerts.tsx already reads user_data/{uid}/aiRecommendation which contains
    the smsText field. The sms_log paths are additional so admins can audit
    every SMS that was ever sent, and so the alerts screen can optionally
    show "Last SMS sent at HH:MM" without needing to know the user UID.
    """
    global last_sms_hash_sent

    if not sms_text or not sms_text.strip():
        print("[SMS] Empty smsText - skipping")
        return

    msg_hash = str(hash(sms_text.strip()))
    if msg_hash == last_sms_hash_sent:
        print("[SMS] Recommendation unchanged - skipping auto-send")
        return

    if not AT_USERNAME or not AT_API_KEY or AT_USERNAME.lower() == "sandbox":
        print("[SMS] Skipping auto-send - not in live mode")
        return

    try:
        users_data = firebase_get("users")
        if not users_data or not isinstance(users_data, dict):
            print("[SMS] No users found in Firebase")
            return

        farmers = [
            {"name": u.get("name", "Farmer"), "phone": u.get("phone", "")}
            for u in users_data.values()
            if isinstance(u, dict)
            and u.get("role", "").lower() == "farmer"
            and u.get("smsEnabled")        == True
            and u.get("phoneVerified")     == True
            and u.get("phone")
        ]

        if not farmers:
            print("[SMS] No SMS-enabled farmers found")
            return

        now        = time.time()
        sent       = 0
        skipped    = 0
        errors     = 0
        recipients = []

        for farmer in farmers:
            phone = clean_phone(farmer["phone"])
            name  = farmer["name"]

            if now - last_sms_sent.get(phone, 0) < SMS_COOLDOWN:
                remaining = int(SMS_COOLDOWN - (now - last_sms_sent.get(phone, 0)))
                skipped += 1
                recipients.append({"name": name, "phone": phone, "status": f"skipped - cooldown {remaining // 60}m"})
                continue

            try:
                message  = sms_text if len(sms_text) <= 320 else sms_text[:317] + "..."
                response = sms_client.send(message, [phone])
                last_sms_sent[phone] = now
                sent += 1
                recipients.append({"name": name, "phone": phone, "status": "sent"})
                print(f"[SMS] Sent to {phone} ({name})")
            except Exception as e:
                errors += 1
                recipients.append({"name": name, "phone": phone, "status": f"error: {str(e)}"})
                print(f"[SMS] Failed for {phone} ({name}): {e}")

        # ── Save to Firebase after sending ───────────────────────────────────
        # This is what allows the app and admin dashboard to see what was sent.
        if sent > 0:
            last_sms_hash_sent = msg_hash
            now_iso = datetime.utcnow().isoformat() + "Z"
            now_ms  = int(time.time() * 1000)

            sms_record = {
                "sentAt":         now_iso,      # human-readable timestamp
                "sentAtMs":       now_ms,        # ms timestamp for sorting
                "message":        sms_text,      # the full SMS text as sent
                "recipientCount": sent,
                "recipients":     recipients,    # who got it, who was skipped
            }

            # 1. Append to history log — one entry per send, never overwritten
            firebase_push("sms_log", sms_record)

            # 2. Overwrite the "latest" node — single fixed path for the app to read
            firebase_set("sms_log_latest", sms_record)

            print(f"[SMS] Record saved to Firebase sms_log and sms_log_latest")

        print(f"[SMS] Done - sent: {sent}, skipped: {skipped}, errors: {errors}")

    except Exception as e:
        print(f"[SMS] Auto-send error: {e}")

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
        threading.Thread(target=sync_and_train, daemon=True).start()
    else:
        sync_and_train()
    threading.Thread(target=_auto_retrain_loop, daemon=True).start()

_startup()

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="SoilApp AI Backend", version="8.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Schemas ──────────────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    soil_moisture:    float = Field(..., ge=0,   le=100)
    soil_temperature: float = Field(..., ge=-10, le=60)
    pest_presence:    int   = Field(..., ge=0,   le=1)
    air_temperature:  float = Field(..., ge=-10, le=50)
    rainfall:         float = Field(..., ge=0)
    humidity:         float = Field(..., ge=0,   le=100)
    wind_speed:       float = Field(..., ge=0)
    sms_text:         Optional[str] = Field(None)

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

class PestDetectInput(BaseModel):
    signal:          List[float] = Field(...)
    sampling_rate:   int         = Field(10000, ge=100,  le=50000)
    sensor_depth_cm: float       = Field(10.0,  ge=0,    le=100)
    top_n:           int         = Field(3,      ge=1,    le=7)

class PestMatch(BaseModel):
    pest: str; score: float; confidence: str; damage: str; treatment: str

class PestDetectResponse(BaseModel):
    dominant_frequency_hz: float; signal_std: float; sensor_depth_cm: float
    matches: List[PestMatch]; summary: str

class OtpRequest(BaseModel):
    phone: str

class OtpVerifyRequest(BaseModel):
    phone: str; otp: str

class FarmerEntry(BaseModel):
    name: str; phone: str

class SmsRequest(BaseModel):
    farmers: List[FarmerEntry]; recommendation: str

# ─── Recommendation builder ───────────────────────────────────────────────────

def build_recommendation(irr: bool, pest: str, planting: str) -> str:
    if irr:
        return "Soil moisture is critically low. Irrigate within 24 hours to prevent crop stress."
    if pest == "high":
        return "High pest risk detected. Apply preventive treatment and inspect crops closely."
    if pest == "medium":
        return "Moderate pest risk. Monitor field edges - conditions favour pest activity."
    if planting == "optimal":
        return "Excellent conditions for planting. Soil and weather are ideal - plant now."
    if planting == "suboptimal":
        return "Conditions acceptable but not ideal. Wait 1-2 days for improvement."
    return "All indicators are safe. No immediate action required. Monitor daily."

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    sms_mode = "not_configured"
    if AT_USERNAME and AT_API_KEY:
        sms_mode = "sandbox" if AT_USERNAME.lower() == "sandbox" else "live"
    return {"status": "ok", "version": "8.1.0", "models_loaded": list(_state["models"].keys()),
            "is_training": _state["meta"]["is_training"], "sms_mode": sms_mode}


@app.get("/model-status")
def model_status():
    return _state["meta"]


@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorInput, background_tasks: BackgroundTasks):
    if not _state["models"] or _state["scaler"] is None:
        raise HTTPException(status_code=503, detail="Models still initialising. Retry shortly.")

    X  = np.array([[data.soil_moisture, data.soil_temperature, data.pest_presence,
                    data.air_temperature, data.rainfall, data.humidity, data.wind_speed]])
    Xs = _state["scaler"].transform(X)

    irr_p  = _state["models"]["irrigation_needed"].predict_proba(Xs)[0]
    irr_c  = int(_state["models"]["irrigation_needed"].predict(Xs)[0])
    pest_p = _state["models"]["pest_risk"].predict_proba(Xs)[0]
    pest_c = int(_state["models"]["pest_risk"].predict(Xs)[0])
    plan_p = _state["models"]["planting_window"].predict_proba(Xs)[0]
    plan_c = int(_state["models"]["planting_window"].predict(Xs)[0])

    pest_label     = PEST_MAP[pest_c]
    planting_label = PLANTING_MAP[plan_c]
    importance     = dict(zip(FEATURES, _state["models"]["irrigation_needed"].feature_importances_.round(3).tolist()))
    recommendation = build_recommendation(bool(irr_c), pest_label, planting_label)

    # Use the rich smsText from the app if provided, otherwise use short fallback
    sms_body = data.sms_text.strip() if data.sms_text and data.sms_text.strip() else recommendation

    # Write short recommendation to shared Firebase path (backward compat)
    firebase_set("soil_monitoring_system/aiRecommendation", {
        "text": recommendation, "timestamp": datetime.utcnow().isoformat() + "Z",
        "irrigation_needed": bool(irr_c), "pest_risk": pest_label, "planting_window": planting_label,
    })

    # Send SMS to all smsEnabled farmers + save record to sms_log in Firebase
    background_tasks.add_task(auto_send_sms_to_farmers, sms_body)

    return PredictionResponse(
        irrigationNeeded=bool(irr_c), irrigationConfidence=float(irr_p[irr_c]),
        pestRisk=pest_label, pestRiskScore=float(pest_p[pest_c]),
        plantingWindow=planting_label, plantingScore=float(plan_p[plan_c]),
        recommendation=recommendation, featureImportance=importance, modelMeta=_state["meta"],
    )


@app.post("/sync-and-retrain")
def force_retrain(background_tasks: BackgroundTasks):
    if _retrain_lock.locked():
        return {"status": "already_running", "message": "Training already in progress."}
    background_tasks.add_task(sync_and_train)
    return {"status": "started", "message": "Firebase sync and retrain started."}


@app.post("/retrain-manual")
def retrain_manual(body: ManualReadings, background_tasks: BackgroundTasks):
    if len(body.readings) < 10:
        raise HTTPException(status_code=400, detail="Need at least 10 readings.")
    def _task():
        df_manual  = _derive_labels(pd.DataFrame(body.readings))
        df_history = fetch_history_df()
        df_seed    = generate_lesotho_seed_data(500)
        frames     = [df_seed, df_manual]
        if not df_history.empty:
            frames.insert(1, df_history)
        train_models(pd.concat(frames, ignore_index=True), history_count=len(df_history) + len(df_manual))
    background_tasks.add_task(_task)
    return {"status": "started", "submitted_readings": len(body.readings)}


@app.get("/feature-importance")
def feature_importance():
    if not _state["models"]:
        raise HTTPException(status_code=503, detail="Models not yet ready.")
    return {name: dict(zip(FEATURES, model.feature_importances_.round(4).tolist()))
            for name, model in _state["models"].items()}


@app.get("/firebase-preview")
def firebase_preview():
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
                parsed[our_field] = _parse_pest_status(raw_val) if our_field == "pest_presence" else raw_val
        preview.append({"firebase_key": key, "raw": entry, "parsed": parsed})
    return {"path": FIREBASE_HISTORY_PATH, "total_records": len(raw), "preview": preview}


@app.get("/sms-log")
def get_sms_log():
    """Returns last 20 SMS records from Firebase for admin review."""
    data = firebase_get("sms_log")
    if not data or not isinstance(data, dict):
        return {"count": 0, "records": []}
    records = sorted(data.values(), key=lambda x: x.get("sentAtMs", 0), reverse=True)
    return {"count": len(records), "records": records[:20]}


@app.post("/detect_pest", response_model=PestDetectResponse)
def detect_pest(data: PestDetectInput):
    if len(data.signal) < 100:
        raise HTTPException(status_code=400, detail=f"Signal too short ({len(data.signal)} samples). Need at least 100.")
    dom_freq   = _dominant_frequency(data.signal, data.sampling_rate)
    signal_std = float(np.std(data.signal))
    matches    = _match_pests(data.signal, data.sampling_rate, top_n=data.top_n)
    if not matches:
        summary = "No significant pest vibration signatures detected in this sample."
    elif matches[0]["confidence"] == "High":
        summary = f"Strong match: {matches[0]['pest']} detected. Immediate inspection recommended."
    elif matches[0]["confidence"] == "Medium":
        summary = f"Possible activity: {matches[0]['pest']}. Monitor closely over next 24-48 hours."
    else:
        summary = "Low-level vibration detected. Could be environmental noise or early-stage activity."
    return PestDetectResponse(dominant_frequency_hz=round(dom_freq, 2), signal_std=round(signal_std, 6),
                               sensor_depth_cm=data.sensor_depth_cm,
                               matches=[PestMatch(**m) for m in matches], summary=summary)


@app.post("/send-otp")
async def send_otp(body: OtpRequest):
    if not AT_USERNAME or not AT_API_KEY:
        return JSONResponse({"success": False, "error": "SMS not configured on server."}, status_code=503)
    try:
        phone = clean_phone(body.phone)
        otp   = str(random.randint(100000, 999999))
        otp_store[phone] = {"otp": otp, "expires": time.time() + 600}
        sms_client.send(f"AGRI ALERT: Your SoilApp verification code is {otp}. Valid for 10 minutes.", [phone])
        return JSONResponse({"success": True, "message": f"OTP sent to {phone}"})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/verify-otp")
async def verify_otp(body: OtpVerifyRequest):
    try:
        phone  = clean_phone(body.phone)
        record = otp_store.get(phone)
        if not record:
            return JSONResponse({"success": False, "error": "No OTP found. Please request a new one."}, status_code=400)
        if time.time() > record["expires"]:
            del otp_store[phone]
            return JSONResponse({"success": False, "error": "OTP expired. Please request a new one."}, status_code=400)
        if record["otp"] != body.otp.strip():
            return JSONResponse({"success": False, "error": "Incorrect OTP. Please try again."}, status_code=400)
        del otp_store[phone]
        return JSONResponse({"success": True, "message": "Phone number verified."})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/send-sms-recommendation")
async def send_sms_recommendation(body: SmsRequest):
    if not AT_USERNAME or not AT_API_KEY:
        return JSONResponse({"success": False, "error": "SMS credentials not configured."}, status_code=503)
    results = []
    now = time.time()
    for farmer in body.farmers:
        phone = clean_phone(farmer.phone)
        if not phone:
            results.append({"phone": "unknown", "status": "skipped - no phone"})
            continue
        if now - last_sms_sent.get(phone, 0) < SMS_COOLDOWN:
            remaining = int(SMS_COOLDOWN - (now - last_sms_sent.get(phone, 0)))
            results.append({"phone": phone, "status": f"skipped - cooldown {remaining // 60}m"})
            continue
        try:
            message = f"AGRI ALERT for {farmer.name}:\n{body.recommendation}\n— SoilApp {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            if len(message) > 320:
                message = message[:317] + "..."
            sms_client.send(message, [phone])
            last_sms_sent[phone] = now
            results.append({"phone": phone, "status": "sent"})
        except Exception as e:
            results.append({"phone": phone, "status": f"error: {str(e)}"})
    return JSONResponse({"success": True, "sent": sum(1 for r in results if r["status"] == "sent"),
                         "skipped": sum(1 for r in results if "skipped" in r["status"]),
                         "errors": sum(1 for r in results if "error" in r["status"]), "results": results})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)