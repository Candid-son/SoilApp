import json
import os
import random
import threading
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import List, Literal, Optional

import africastalking
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

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
    print("⚠️  [SMS] WARNING: AT_USERNAME or AT_API_KEY is not set — SMS will fail.")
elif AT_USERNAME.lower() == "sandbox":
    print("⚠️  [SMS] WARNING: AT_USERNAME is 'sandbox' — SMS will only work in the AT simulator.")
else:
    print(f"✅ [SMS] Africa's Talking initialised in LIVE mode (username: {AT_USERNAME})")

africastalking.initialize(username=AT_USERNAME, api_key=AT_API_KEY)
sms_client = africastalking.SMS

# ─── SMS config ───────────────────────────────────────────────────────────────
SMS_COOLDOWN          = 3600
STARTUP_GRACE_SECONDS = 120
_server_start_time    = time.time()

last_sms_sent: dict      = {}
last_recommendation_sent = ""
last_recommendation_ts   = ""

AI_REC_POLL_INTERVAL  = 10
SENSOR_POLL_INTERVAL  = 60

otp_store: dict = {}

# ─── Normal-range thresholds ──────────────────────────────────────────────────

MOISTURE_NORMAL_LOW   = 30
MOISTURE_NORMAL_HIGH  = 85
SOIL_TEMP_NORMAL_LOW  = 5
SOIL_TEMP_NORMAL_HIGH = 38
AIR_TEMP_NORMAL_LOW   = 10
AIR_TEMP_NORMAL_HIGH  = 28
HUMIDITY_NORMAL_HIGH  = 70
RAINFALL_LOW          = 2.0
PEST_SCORE_MEDIUM     = 0.30
PEST_SCORE_HIGH       = 0.60

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

NO_PEST_PHRASES  = {"no pest", "clear", "none", "normal", "no pest detected", "no pest activity"}
PEST_MAP         = {0: "low",   1: "medium",     2: "high"}
PLANTING_MAP     = {0: "avoid", 1: "suboptimal", 2: "optimal"}

# ═══════════════════════════════════════════════════════════════════════════════
#  SOIL PEST VIBRATION IDENTIFICATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PestProfile:
    name: str
    common_name: str
    freq_min: float
    freq_max: float
    peak_freq_low: float
    peak_freq_high: float
    burst_rate_min: float
    burst_rate_max: float
    amp_min: float
    amp_max: float
    pattern: str
    damage_type: str
    treatment: str


@dataclass
class IdentificationResult:
    rank: int
    pest: PestProfile
    score: float
    confidence: str
    peak_freq_hz: float
    burst_rate_per_sec: float
    amplitude_db: float
    temporal_pattern: str


VIBRATION_PROFILES: List[PestProfile] = [
    PestProfile(name="Phyllophaga_spp", common_name="White Grub (Scarab Beetle Larva)", freq_min=200, freq_max=800, peak_freq_low=300, peak_freq_high=600, burst_rate_min=0.1, burst_rate_max=1.0, amp_min=20, amp_max=45, pattern="burst", damage_type="Root feeding – chews through grass and crop roots", treatment="Beneficial nematodes (Heterorhabditis bacteriophora), imidacloprid drench"),
    PestProfile(name="Scapteriscus_spp", common_name="Mole Cricket", freq_min=300, freq_max=1500, peak_freq_low=500, peak_freq_high=1000, burst_rate_min=5.0, burst_rate_max=20.0, amp_min=35, amp_max=60, pattern="continuous", damage_type="Tunnelling destroys root systems and uproots seedlings", treatment="Steinernema scapterisci nematodes, bifenthrin baits"),
    PestProfile(name="Agriotes_spp", common_name="Wireworm (Click Beetle Larva)", freq_min=500, freq_max=1500, peak_freq_low=700, peak_freq_high=1200, burst_rate_min=0.05, burst_rate_max=0.5, amp_min=15, amp_max=35, pattern="burst", damage_type="Tunnels through seeds, roots, and underground stems", treatment="Steinernema carpocapsae nematodes, spinosad soil drench"),
    PestProfile(name="Agrotis_spp", common_name="Cutworm (Noctuid Moth Larva)", freq_min=100, freq_max=600, peak_freq_low=150, peak_freq_high=400, burst_rate_min=0.1, burst_rate_max=0.5, amp_min=18, amp_max=38, pattern="burst", damage_type="Severs seedling stems at soil level; surface nocturnal feeder", treatment="Bacillus thuringiensis (Bt), Steinernema carpocapsae nematodes"),
    PestProfile(name="Diaprepes_Otiorhynchus", common_name="Root Weevil Larva", freq_min=400, freq_max=1200, peak_freq_low=500, peak_freq_high=900, burst_rate_min=0.2, burst_rate_max=2.0, amp_min=25, amp_max=50, pattern="burst", damage_type="Girdling of roots; kills citrus and ornamental plants", treatment="Heterorhabditis bacteriophora nematodes, entomopathogenic fungi"),
    PestProfile(name="Reticulitermes_spp", common_name="Subterranean Termite", freq_min=50, freq_max=500, peak_freq_low=100, peak_freq_high=300, burst_rate_min=1.0, burst_rate_max=10.0, amp_min=30, amp_max=55, pattern="rhythmic", damage_type="Destroys woody plant roots and below-ground stems", treatment="Fipronil soil barrier, beneficial nematodes, borate baits"),
    PestProfile(name="Diabrotica_spp", common_name="Corn Rootworm (Chrysomelid Larva)", freq_min=300, freq_max=900, peak_freq_low=400, peak_freq_high=700, burst_rate_min=0.1, burst_rate_max=1.0, amp_min=20, amp_max=42, pattern="burst", damage_type="Prunes corn root system, causing lodging and yield loss", treatment="Crop rotation, Bt CrylF/Cry3 traits, soil-applied insecticides"),
    PestProfile(name="Bradysia_spp", common_name="Fungus Gnat Larva", freq_min=50, freq_max=300, peak_freq_low=80, peak_freq_high=200, burst_rate_min=0.05, burst_rate_max=0.3, amp_min=10, amp_max=25, pattern="burst", damage_type="Feeds on roots and organic matter; damages seedlings", treatment="Steinernema feltiae nematodes, hydrogen peroxide drench"),
]

# ─── Signal processing helpers ────────────────────────────────────────────────

def _bandpass_filter(signal: np.ndarray, fs: float, lowcut: float = 50.0, highcut: float = 5000.0) -> np.ndarray:
    nyq  = 0.5 * fs
    low  = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(4, [low, high], btype="band")
    return filtfilt(b, a, signal)

def _amplitude_db(signal: np.ndarray, ref: float = 1e-6) -> float:
    rms = np.sqrt(np.mean(signal ** 2))
    return 20 * np.log10(rms / ref) if rms > 0 else -np.inf

def _detect_bursts(signal: np.ndarray, fs: float, window_ms: float = 50.0, threshold_factor: float = 3.0):
    window   = max(1, int((window_ms / 1000.0) * fs))
    n_win    = len(signal) // window
    energy   = np.array([np.sqrt(np.mean(signal[i*window:(i+1)*window] ** 2)) for i in range(n_win)])
    threshold  = np.median(energy) * threshold_factor
    burst_idx  = np.where(energy > threshold)[0]
    duration_sec = len(signal) / fs
    burst_rate = len(burst_idx) / duration_sec if duration_sec > 0 else 0.0
    amps = [_amplitude_db(signal[i*window:(i+1)*window]) for i in burst_idx]
    mean_amp = float(np.mean(amps)) if amps else -np.inf
    return burst_idx, burst_rate, mean_amp

def _compute_spectrum(signal: np.ndarray, fs: float, n_fft: int = 4096):
    padded   = np.zeros(n_fft)
    length   = min(len(signal), n_fft)
    padded[:length] = signal[:length]
    windowed = padded * np.hanning(n_fft)
    spectrum = np.abs(fft(windowed))
    freqs    = fftfreq(n_fft, d=1.0 / fs)
    pos      = freqs > 0
    freqs, power = freqs[pos], spectrum[pos] ** 2
    if len(power) > 0:
        peaks, _ = find_peaks(power, height=np.max(power) * 0.1)
        dom_freq = float(freqs[peaks[np.argmax(power[peaks])]] if len(peaks) > 0 else freqs[np.argmax(power)])
    else:
        dom_freq = 0.0
    return freqs, power, dom_freq

def _band_energy(freqs, power, lo, hi):
    return float(np.sum(power[(freqs >= lo) & (freqs <= hi)]))

def _classify_pattern(burst_idx: np.ndarray, burst_rate: float) -> str:
    if burst_rate > 4.0: return "continuous"
    if len(burst_idx) < 4: return "burst"
    ibi = np.diff(burst_idx).astype(float)
    cv  = np.std(ibi) / (np.mean(ibi) + 1e-9)
    return "rhythmic" if cv < 0.3 else "burst"

def _is_background_noise(freqs, power, amplitude_db: float) -> bool:
    lo  = _band_energy(freqs, power, 50, 400)
    hi  = _band_energy(freqs, power, 400, 5000) + 1e-12
    return (lo / hi) > 3.0 and amplitude_db > 65.0

def _score_profile(profile: PestProfile, peak_freq, burst_rate, amplitude_db, temporal_pattern, freqs, power) -> float:
    score = 0.0
    if profile.freq_min <= peak_freq <= profile.freq_max:
        freq_score = 40.0 if profile.peak_freq_low <= peak_freq <= profile.peak_freq_high else 28.0
    else:
        nearest = min(abs(peak_freq - profile.freq_min), abs(peak_freq - profile.freq_max))
        freq_score = 15.0 * max(0, 1 - nearest / (profile.freq_max - profile.freq_min))
    energy_ratio = _band_energy(freqs, power, profile.freq_min, profile.freq_max) / (_band_energy(freqs, power, 50, 5000) + 1e-12)
    freq_score = min(40.0, freq_score * (0.5 + energy_ratio))
    score += freq_score
    if profile.burst_rate_min <= burst_rate <= profile.burst_rate_max:
        score += 30.0
    else:
        ratio = (burst_rate / (profile.burst_rate_min + 1e-9) if burst_rate < profile.burst_rate_min else profile.burst_rate_max / (burst_rate + 1e-9))
        score += 30.0 * max(0, min(1, ratio))
    if profile.amp_min <= amplitude_db <= profile.amp_max:
        score += 20.0
    else:
        diff = (profile.amp_min - amplitude_db if amplitude_db < profile.amp_min else amplitude_db - profile.amp_max)
        score += max(0, 20.0 - diff * 1.5)
    if temporal_pattern == profile.pattern: score += 10.0
    elif {temporal_pattern, profile.pattern} <= {"burst", "rhythmic"}: score += 4.0
    return round(score, 2)

def identify_pest_signal(raw_signal: np.ndarray, sampling_rate: float = 44100.0, top_n: int = 3, min_duration_sec: float = 5.0) -> List[IdentificationResult]:
    duration = len(raw_signal) / sampling_rate
    if duration < min_duration_sec: raise ValueError(f"Signal too short ({duration:.1f}s). Minimum is {min_duration_sec}s.")
    if sampling_rate < 10_000: raise ValueError("Sampling rate must be >= 10,000 Hz.")
    filtered = _bandpass_filter(raw_signal, sampling_rate)
    burst_idx, burst_rate, amplitude_db = _detect_bursts(filtered, sampling_rate)
    if len(burst_idx) == 0: return []
    win = int(0.05 * sampling_rate)
    seg = filtered[burst_idx[0] * win: (burst_idx[0] + 1) * win]
    freqs, power, peak_freq = _compute_spectrum(seg, sampling_rate)
    pattern = _classify_pattern(burst_idx, burst_rate)
    if _is_background_noise(freqs, power, amplitude_db):
        print("[PestDetect] Signal resembles background noise — skipping.")
        return []
    scored = sorted([(p, _score_profile(p, peak_freq, burst_rate, amplitude_db, pattern, freqs, power)) for p in VIBRATION_PROFILES], key=lambda x: x[1], reverse=True)
    results = []
    for rank, (profile, score) in enumerate(scored[:top_n], 1):
        confidence = ("High" if score >= 70 else "Medium" if score >= 45 else "Low" if score >= 25 else "Insufficient signal")
        results.append(IdentificationResult(rank=rank, pest=profile, score=score, confidence=confidence, peak_freq_hz=round(peak_freq, 1), burst_rate_per_sec=round(burst_rate, 3), amplitude_db=round(amplitude_db, 1), temporal_pattern=pattern))
    return results

def generate_vibration_report(results: List[IdentificationResult], sensor_depth_cm: float = 10.0) -> str:
    if not results: return "No significant insect vibration detected."
    r0 = results[0]
    lines = [f"Sensor depth: {sensor_depth_cm} cm | Peak freq: {r0.peak_freq_hz} Hz | Burst rate: {r0.burst_rate_per_sec}/s | Amplitude: {r0.amplitude_db} dB | Pattern: {r0.temporal_pattern}", "TOP CANDIDATES:"]
    for r in results:
        icon = {"High": "✔", "Medium": "◈", "Low": "?"}.get(r.confidence, "✗")
        lines.append(f"#{r.rank} {icon} {r.pest.common_name} — score {r.score}/100 ({r.confidence}) | {r.pest.damage_type} | Tx: {r.pest.treatment}")
    return "\n".join(lines)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def clean_phone(phone: str) -> str:
    phone = phone.strip().replace(" ", "").replace("-", "")
    if phone.startswith("00266"): return "+" + phone[2:]
    if phone.startswith("266") and not phone.startswith("+"): return "+" + phone
    if not phone.startswith("+"): return "+266" + phone
    return phone

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
        if resp.status_code == 200: return resp.json()
        print(f"[Firebase] GET /{path} → HTTP {resp.status_code}: {resp.text[:300]}")
        return None
    except Exception as exc:
        print(f"[Firebase] Connection error: {exc}")
        return None

def firebase_set(path: str, data: dict) -> bool:
    url    = f"{FIREBASE_URL}/{path}.json"
    params = {"auth": FIREBASE_AUTH_TOKEN} if FIREBASE_AUTH_TOKEN else {}
    try:
        resp = requests.put(url, json=data, params=params, timeout=10)
        return resp.status_code == 200
    except Exception as exc:
        print(f"[Firebase] Write error: {exc}")
        return False

def _parse_pest_status(value) -> int:
    if isinstance(value, (int, float)): return int(bool(value))
    if isinstance(value, str):
        cleaned = value.strip().lower()
        if cleaned in NO_PEST_PHRASES or "no pest" in cleaned or cleaned == "clear": return 0
        return 1
    return 0

def fetch_history_df() -> pd.DataFrame:
    raw = firebase_get(FIREBASE_HISTORY_PATH)
    if not raw or not isinstance(raw, dict): return pd.DataFrame()
    rows, total_raw = [], 0
    for entry in raw.values():
        if not isinstance(entry, dict): continue
        total_raw += 1
        row: dict = {}
        for fb_field, our_field in FIELD_ALIASES.items():
            if fb_field in entry and our_field not in row:
                raw_val = entry[fb_field]
                row[our_field] = (_parse_pest_status(raw_val) if our_field == "pest_presence" else raw_val)
        rows.append(row)
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col, default in LESOTHO_MEANS.items():
        if col not in df.columns: df[col] = default
        else: df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)
    for col in FEATURES:
        if col not in df.columns: df[col] = np.nan
        else: df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["soil_moisture", "soil_temperature", "pest_presence"], inplace=True)
    df.dropna(subset=FEATURES, inplace=True)
    valid = (df["soil_temperature"] >= -10) & (df["soil_temperature"] <= 60) & (df["soil_moisture"] >= 0) & (df["soil_moisture"] <= 100)
    df = df[valid].copy()
    skipped = total_raw - len(df)
    if df.empty: return df
    df = _derive_labels(df)
    _state["meta"]["firebase_connected"] = True
    _state["meta"]["skipped_rows"] = skipped
    print(f"[Firebase] Loaded {len(df)} usable rows (skipped {skipped} of {total_raw}).")
    return df

def _derive_labels(df: pd.DataFrame) -> pd.DataFrame:
    sm, st, pp = df["soil_moisture"].values, df["soil_temperature"].values, df["pest_presence"].values
    at, rf, hum = df["air_temperature"].values, df["rainfall"].values, df["humidity"].values
    df["irrigation_needed"] = ((sm < MOISTURE_NORMAL_LOW) & (rf < RAINFALL_LOW)).astype(int)
    pest_score = (pp * 0.50 + (hum > HUMIDITY_NORMAL_HIGH).astype(float) * 0.25 + (st > 25).astype(float) * 0.15 + (at > AIR_TEMP_NORMAL_HIGH).astype(float) * 0.10)
    df["pest_risk"] = np.where(pest_score >= PEST_SCORE_HIGH, 2, np.where(pest_score >= PEST_SCORE_MEDIUM, 1, 0))
    planting_score = (((sm >= MOISTURE_NORMAL_LOW) & (sm <= MOISTURE_NORMAL_HIGH)).astype(float) * 0.35 + ((st >= SOIL_TEMP_NORMAL_LOW) & (st <= 25)).astype(float) * 0.30 + (rf < 3).astype(float) * 0.20 + ((at >= AIR_TEMP_NORMAL_LOW) & (at <= AIR_TEMP_NORMAL_HIGH)).astype(float) * 0.15)
    df["planting_window"] = np.where(planting_score > 0.70, 2, np.where(planting_score > 0.40, 1, 0))
    return df

def generate_lesotho_seed_data(n: int = 2000) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"soil_moisture": rng.uniform(5, 95, n), "soil_temperature": rng.uniform(5, 38, n), "pest_presence": rng.choice([0, 1], n, p=[0.65, 0.35]), "air_temperature": rng.uniform(2, 34, n), "rainfall": rng.exponential(2.5, n).clip(0, 40), "humidity": rng.uniform(25, 95, n), "wind_speed": rng.uniform(0, 40, n)})
    return _derive_labels(df)

def train_models(df: pd.DataFrame, history_count: int = 0):
    _state["meta"]["is_training"] = True
    print(f"\nTraining on {len(df)} samples ({history_count} from Firebase)...")
    X, scaler = df[FEATURES].values, StandardScaler()
    Xs = scaler.fit_transform(X)
    models, accuracies = {}, {}
    for target in TARGETS:
        y = df[target].values
        try: Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
        except ValueError: Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5, class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        acc = float(accuracy_score(yte, clf.predict(Xte)))
        models[target], accuracies[target] = clf, round(acc, 4)
        print(f"  [{target}] accuracy = {acc:.3f}")
    joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
    for name, model in models.items(): joblib.dump(model, f"{MODEL_DIR}/{name}.pkl")
    _state["models"], _state["scaler"] = models, scaler
    _state["meta"].update({"last_trained": datetime.utcnow().isoformat() + "Z", "seed_samples": len(df) - history_count, "history_samples": history_count, "total_samples": len(df), "is_training": False, "accuracy": accuracies})
    with open(f"{MODEL_DIR}/meta.json", "w") as fh: json.dump(_state["meta"], fh, indent=2)
    print(f"Training complete — {len(df)} total, {history_count} from Firebase.\n")

def sync_and_train():
    with _retrain_lock:
        df_history = fetch_history_df()
        history_count = len(df_history)
        seed_n = max(200, 2000 - history_count * 3)
        df_seed = generate_lesotho_seed_data(seed_n)
        if history_count >= MIN_HISTORY_FOR_BLEND: df_combined = pd.concat([df_seed, df_history], ignore_index=True)
        else: df_combined = df_seed
        train_models(df_combined, history_count=history_count)

def _auto_retrain_loop():
    while True:
        time.sleep(RETRAIN_INTERVAL_HOURS * 3600)
        print(f"\nScheduled retrain (every {RETRAIN_INTERVAL_HOURS}h)")
        sync_and_train()

# ─── VIBRATION WATCHER ────────────────────────────────────────────────────────

VIBRATION_DEVICE_PATH   = "soil_monitoring_system/sensors/device001"
VIBRATION_POLL_INTERVAL = 15

def _reassemble_chunks(chunks_data: dict, chunk_size: int, total_chunks: int) -> np.ndarray:
    signal = []
    for chunk_idx in range(total_chunks):
        key = f"chunk_{chunk_idx}"
        chunk = chunks_data.get(key)
        if not chunk or not isinstance(chunk, dict): signal.extend([0.0] * chunk_size); continue
        ordered = [float(chunk[str(i)]) for i in range(len(chunk)) if str(i) in chunk]
        signal.extend(ordered)
    return np.array(signal, dtype=float)

def _process_vibration_from_firebase():
    device_path = VIBRATION_DEVICE_PATH
    print("[VibWatch] ready=true detected — starting analysis...")
    meta = firebase_get(f"{device_path}/vibrationMeta")
    if not meta or not isinstance(meta, dict): print("[VibWatch] vibrationMeta missing — aborting"); return
    sample_rate = int(meta.get("sampleRate", 10000))
    chunk_size = int(meta.get("chunkSize", 2000))
    total_chunks = int(meta.get("totalChunks", 50))
    sensor_depth = float(firebase_get(f"{device_path}/sensorDepthCm") or 10.0)
    firebase_set(f"{device_path}/vibrationMeta", {**meta, "ready": False})
    chunks_data = firebase_get(f"{device_path}/vibrationSamples")
    if not chunks_data or not isinstance(chunks_data, dict): _write_vibration_error("No vibration sample chunks found."); return
    signal = _reassemble_chunks(chunks_data, chunk_size, total_chunks)
    if len(signal) < sample_rate * 5: _write_vibration_error(f"Signal too short ({len(signal)} samples)."); return
    try: results = identify_pest_signal(raw_signal=signal, sampling_rate=float(sample_rate), top_n=3, min_duration_sec=5.0)
    except Exception as e: _write_vibration_error(str(e)); return
    report = generate_vibration_report(results, sensor_depth_cm=sensor_depth)
    matches_payload = [{"rank": r.rank, "pest": r.pest.common_name, "scientificName": r.pest.name.replace("_", " "), "score": r.score, "confidence": r.confidence, "peakFreqHz": r.peak_freq_hz, "burstRatePerSec": r.burst_rate_per_sec, "amplitudeDb": r.amplitude_db, "temporalPattern": r.temporal_pattern, "damageType": r.pest.damage_type, "treatment": r.pest.treatment} for r in results]
    if not results: summary = "No significant insect vibration detected."
    elif results[0].confidence == "High": summary = f"Strong match: {results[0].pest.common_name} detected (score {results[0].score}/100)."
    elif results[0].confidence == "Medium": summary = f"Possible activity: {results[0].pest.common_name} (score {results[0].score}/100)."
    else: summary = f"Weak signal — possible {results[0].pest.common_name}."
    payload = {"timestamp": datetime.utcnow().isoformat() + "Z", "sampleRate": sample_rate, "signalSamples": len(signal), "signalDurationSec": round(len(signal) / sample_rate, 2), "sensorDepthCm": sensor_depth, "summary": summary, "report": report, "matches": matches_payload, "error": None, "analysedAt": datetime.utcnow().isoformat() + "Z"}
    ok = firebase_set(f"{device_path}/vibrationAnalysis", payload)
    if ok: print(f"[VibWatch] ✅ Results written to Firebase | {summary[:80]}")
    else: print("[VibWatch] ❌ Failed to write results to Firebase")

def _write_vibration_error(msg: str):
    firebase_set(f"{VIBRATION_DEVICE_PATH}/vibrationAnalysis", {"timestamp": datetime.utcnow().isoformat() + "Z", "summary": f"Analysis failed: {msg}", "report": msg, "matches": [], "error": msg})

def _vibration_watcher_loop():
    print(f"[VibWatch] Watcher started — polling every {VIBRATION_POLL_INTERVAL}s")
    while True:
        try:
            meta = firebase_get(f"{VIBRATION_DEVICE_PATH}/vibrationMeta")
            if meta and isinstance(meta, dict) and meta.get("ready") is True: _process_vibration_from_firebase()
        except Exception as e: print(f"[VibWatch] Poll error: {e}")
        time.sleep(VIBRATION_POLL_INTERVAL)

# ─── SENSOR WATCHER ───────────────────────────────────────────────────────────

def _validate_sensor_data(data: dict) -> Optional[dict]:
    moisture = data.get("moisture") or data.get("soilMoisture") or data.get("soil_moisture")
    temp = data.get("temperature") or data.get("soilTemperature") or data.get("soil_temperature")
    if moisture is None or temp is None: return None
    try: moisture = float(moisture); temp = float(temp)
    except (TypeError, ValueError): return None
    if not (0 <= moisture <= 100) or not (-10 <= temp <= 60): return None
    pest_raw = data.get("pest_status") or data.get("pestPresence") or data.get("pest_presence") or 0
    air_temp = float(data.get("airTemperature") or data.get("air_temperature") or LESOTHO_MEANS["air_temperature"])
    rainfall = float(data.get("rainfall") or LESOTHO_MEANS["rainfall"])
    humidity = float(data.get("humidity") or LESOTHO_MEANS["humidity"])
    wind_speed = float(data.get("windSpeed") or data.get("wind_speed") or LESOTHO_MEANS["wind_speed"])
    return {"soil_moisture": moisture, "soil_temperature": temp, "pest_presence": _parse_pest_status(pest_raw), "air_temperature": max(-10, min(50, air_temp)), "rainfall": max(0, rainfall), "humidity": max(0, min(100, humidity)), "wind_speed": max(0, wind_speed)}

def _conditions_warrant_sms(irr: bool, pest: str, planting: str, sensor: Optional[dict] = None) -> bool:
    if irr:
        sm = sensor.get("soil_moisture", 100) if sensor else 100
        rf = sensor.get("rainfall", 10) if sensor else 10
        if sm < MOISTURE_NORMAL_LOW and rf < RAINFALL_LOW: return True
    if pest in ("medium", "high"): return True
    if planting == "optimal" and not irr and pest == "low": return True
    return False

def _run_prediction_from_sensor(clean: dict) -> bool:
    if not _state["models"] or _state["scaler"] is None: return False
    X = np.array([[clean["soil_moisture"], clean["soil_temperature"], clean["pest_presence"], clean["air_temperature"], clean["rainfall"], clean["humidity"], clean["wind_speed"]]])
    Xs = _state["scaler"].transform(X)
    irr_c = int(_state["models"]["irrigation_needed"].predict(Xs)[0])
    pest_c = int(_state["models"]["pest_risk"].predict(Xs)[0])
    plan_c = int(_state["models"]["planting_window"].predict(Xs)[0])
    pest_label = PEST_MAP[pest_c]
    planting_label = PLANTING_MAP[plan_c]
    recommendation = build_recommendation(bool(irr_c), pest_label, planting_label, clean)
    if not _conditions_warrant_sms(bool(irr_c), pest_label, planting_label, clean): return True
    ok = firebase_set("soil_monitoring_system/aiRecommendation", {"text": recommendation, "timestamp": datetime.utcnow().isoformat() + "Z", "irrigation_needed": bool(irr_c), "pest_risk": pest_label, "planting_window": planting_label})
    if ok: print(f"[SensorWatch] ✅ aiRecommendation updated | {recommendation[:80]}")
    return ok

def _sensor_watcher_loop():
    print(f"[SensorWatch] Started — polling every {SENSOR_POLL_INTERVAL}s")
    while True:
        try:
            data = firebase_get(VIBRATION_DEVICE_PATH)
            if data and isinstance(data, dict):
                clean = _validate_sensor_data(data)
                if clean: _run_prediction_from_sensor(clean)
        except Exception as e: print(f"[SensorWatch] Error: {e}")
        time.sleep(SENSOR_POLL_INTERVAL)

# ─── HOURLY AVERAGE WATCHER ───────────────────────────────────────────────────

HOURLY_HISTORY_PATH = "soil_monitoring_system/hourlyHistory"
_hourly_buffer: list = []
_current_hour: int = -1

def _flush_hourly_buffer(hour_label: str):
    global _hourly_buffer
    if not _hourly_buffer: return
    avg_temp = round(sum(r["temperature"] for r in _hourly_buffer) / len(_hourly_buffer), 1)
    avg_moisture = round(sum(r["moisture"] for r in _hourly_buffer) / len(_hourly_buffer), 1)
    entry = {"timestamp": datetime.utcnow().isoformat() + "Z", "hourLabel": hour_label, "temperature": avg_temp, "moisture": avg_moisture, "samples": len(_hourly_buffer)}
    safe_key = hour_label.replace(" ", "_").replace(":", "-")
    firebase_set(f"{HOURLY_HISTORY_PATH}/{safe_key}", entry)
    _hourly_buffer = []

def _hourly_average_loop():
    global _current_hour
    while True:
        try:
            data = firebase_get(VIBRATION_DEVICE_PATH)
            if data and isinstance(data, dict):
                clean = _validate_sensor_data(data)
                if clean:
                    _hourly_buffer.append({"temperature": clean["soil_temperature"], "moisture": clean["soil_moisture"]})
                    now = datetime.utcnow()
                    hour = now.hour
                    if _current_hour == -1: _current_hour = hour
                    elif hour != _current_hour:
                        hour_label = now.strftime("%Y-%m-%d") + f" {_current_hour:02d}:00"
                        _flush_hourly_buffer(hour_label)
                        _current_hour = hour
        except Exception as e: print(f"[HourlyWatch] Error: {e}")
        time.sleep(60)

# ─── AUTO SMS ─────────────────────────────────────────────────────────────────

def _sms_cooldown_key(phone: str, alert_type: str) -> str:
    return f"{phone}::{alert_type}"

def _is_sms_on_cooldown(phone: str, alert_type: str, now: float) -> bool:
    key = _sms_cooldown_key(phone, alert_type)
    last_sent = last_sms_sent.get(key, 0)
    return (now - last_sent) < SMS_COOLDOWN

def _mark_sms_sent(phone: str, alert_type: str, now: float):
    last_sms_sent[_sms_cooldown_key(phone, alert_type)] = now

def _derive_alert_type(recommendation: str) -> str:
    rec = recommendation.lower()
    if "irrigat" in rec: return "irrigation"
    if "pest" in rec: return "pest"
    if "plant" in rec: return "planting"
    return "general"

def auto_send_sms_to_farmers(_unused: str = ""):
    global last_recommendation_sent

    # ── Startup grace period ──────────────────────────────────────────────────
    elapsed = time.time() - _server_start_time
    if elapsed < STARTUP_GRACE_SECONDS:
        remaining = int(STARTUP_GRACE_SECONDS - elapsed)
        print(f"[SMS] Startup grace period active — blocking SMS for {remaining}s more")
        return

    # ── CHECK ADMIN SETTINGS: SMS Alerts toggle ───────────────────────────────
    admin_settings = firebase_get("soil_monitoring_system/admin_settings")
    if admin_settings and admin_settings.get("smsAlerts") is False:
        print("[SMS] ⏸ SMS alerts disabled by admin — skipping auto-send")
        return

    if not AT_USERNAME or not AT_API_KEY or AT_USERNAME.lower() == "sandbox":
        print("[SMS] Skipping auto-send — not in live mode")
        return

    ai_rec = firebase_get("soil_monitoring_system/aiRecommendation")
    if not ai_rec or not isinstance(ai_rec, dict):
        print("[SMS] No aiRecommendation node found in Firebase — skipping")
        return

    recommendation = ai_rec.get("text", "").strip()
    if not recommendation:
        print("[SMS] aiRecommendation/text is empty — skipping")
        return

    if recommendation == last_recommendation_sent:
        print("[SMS] Recommendation unchanged — skipping auto-send")
        return

    alert_type = _derive_alert_type(recommendation)
    print(f"[SMS] New recommendation (type: {alert_type}): {recommendation[:80]}...")

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
            and u.get("smsEnabled") == True
            and u.get("phoneVerified") == True
            and u.get("phone")
        ]

        if not farmers:
            print("[SMS] No SMS-enabled farmers found")
            return

        now, sent, skipped, errors = time.time(), 0, 0, 0

        for farmer in farmers:
            phone = clean_phone(farmer["phone"])
            name = farmer["name"]

            if _is_sms_on_cooldown(phone, alert_type, now):
                skipped += 1
                continue

            try:
                message = f"AGRI ALERT for {name}:\n{recommendation}\n— SoilApp {datetime.now().strftime('%d/%m/%Y %H:%M')}"
                if len(message) > 320: message = message[:317] + "..."
                response = sms_client.send(message, [phone])
                _mark_sms_sent(phone, alert_type, now)
                sent += 1
                print(f"[SMS] ✅ Sent to {phone} ({name}) | {response}")
            except Exception as e:
                errors += 1
                print(f"[SMS] ❌ Failed for {phone} ({name}): {e}")

        if sent > 0: last_recommendation_sent = recommendation
        print(f"[SMS] Done — sent: {sent}, skipped: {skipped}, errors: {errors}")

    except Exception as e:
        print(f"[SMS] Auto-send error: {e}")

# ─── AI RECOMMENDATION WATCHER ───────────────────────────────────────────────

def _ai_recommendation_watcher_loop():
    global last_recommendation_ts
    print(f"[RecWatch] Watcher started — polling every {AI_REC_POLL_INTERVAL}s")
    while True:
        try:
            ai_rec = firebase_get("soil_monitoring_system/aiRecommendation")
            if ai_rec and isinstance(ai_rec, dict):
                new_ts = ai_rec.get("timestamp", "")
                new_text = ai_rec.get("text", "").strip()
                if not new_ts or not new_text or new_ts == last_recommendation_ts: pass
                elif new_text == last_recommendation_sent:
                    last_recommendation_ts = new_ts
                else:
                    print(f"[RecWatch] New recommendation (ts: {new_ts})")
                    last_recommendation_ts = new_ts
                    auto_send_sms_to_farmers()
        except Exception as e: print(f"[RecWatch] Poll error: {e}")
        time.sleep(AI_REC_POLL_INTERVAL)

# ─── Startup ──────────────────────────────────────────────────────────────────

def _startup():
    model_paths = [f"{MODEL_DIR}/scaler.pkl"] + [f"{MODEL_DIR}/{t}.pkl" for t in TARGETS]
    if all(os.path.exists(p) for p in model_paths):
        _state["scaler"] = joblib.load(f"{MODEL_DIR}/scaler.pkl")
        _state["models"] = {t: joblib.load(f"{MODEL_DIR}/{t}.pkl") for t in TARGETS}
        meta_path = f"{MODEL_DIR}/meta.json"
        if os.path.exists(meta_path):
            with open(meta_path) as fh: _state["meta"].update(json.load(fh))
        print("Models loaded. Syncing Firebase history in background...")
        threading.Thread(target=sync_and_train, daemon=True).start()
    else:
        print("First run — training on seed + Firebase history...")
        sync_and_train()
    threading.Thread(target=_auto_retrain_loop, daemon=True).start()
    threading.Thread(target=_vibration_watcher_loop, daemon=True).start()
    threading.Thread(target=_ai_recommendation_watcher_loop, daemon=True).start()
    threading.Thread(target=_sensor_watcher_loop, daemon=True).start()
    threading.Thread(target=_hourly_average_loop, daemon=True).start()

_startup()

# ─── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(title="SoilApp AI Backend", description="Random Forest models + vibration pest detection + Firebase sensor watcher + hourly history + auto SMS for Lesotho farmers", version="9.5.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ─── Schemas ──────────────────────────────────────────────────────────────────

class SensorInput(BaseModel):
    soil_moisture: float = Field(..., ge=0, le=100)
    soil_temperature: float = Field(..., ge=-10, le=60)
    pest_presence: int = Field(..., ge=0, le=1)
    air_temperature: float = Field(..., ge=-10, le=50)
    rainfall: float = Field(..., ge=0)
    humidity: float = Field(..., ge=0, le=100)
    wind_speed: float = Field(..., ge=0)

class PredictionResponse(BaseModel):
    irrigationNeeded: bool
    irrigationConfidence: float
    pestRisk: Literal["low", "medium", "high"]
    pestRiskScore: float
    plantingWindow: Literal["optimal", "suboptimal", "avoid"]
    plantingScore: float
    recommendation: str
    featureImportance: dict
    modelMeta: dict

class ManualReadings(BaseModel):
    readings: list[dict]

class PestDetectInput(BaseModel):
    signal: List[float] = Field(..., description="Vibration samples in m/s² from soil sensor")
    sampling_rate: float = Field(44100.0, ge=10000, le=100000)
    sensor_depth_cm: float = Field(10.0, ge=0, le=100)
    top_n: int = Field(3, ge=1, le=8)
    min_duration_sec: float = Field(5.0, ge=1.0)

class PestMatchResult(BaseModel):
    rank: int; pest: str; scientific_name: str; score: float; confidence: str
    peak_freq_hz: float; burst_rate_per_sec: float; amplitude_db: float
    temporal_pattern: str; damage_type: str; treatment: str

class PestDetectResponse(BaseModel):
    matches: List[PestMatchResult]; summary: str; report: str
    sensor_depth_cm: float; signal_duration_sec: float

class OtpRequest(BaseModel):
    phone: str

class OtpVerifyRequest(BaseModel):
    phone: str; otp: str

class FarmerEntry(BaseModel):
    name: str; phone: str

class SmsRequest(BaseModel):
    farmers: List[FarmerEntry]; recommendation: str

# ─── build_recommendation ─────────────────────────────────────────────────────

def build_recommendation(irr: bool, pest: str, planting: str, sensor: Optional[dict] = None) -> str:
    parts = []
    if irr:
        extra = ""
        if sensor:
            sm = sensor.get("soil_moisture", 0)
            st = sensor.get("soil_temperature", 0)
            if st > AIR_TEMP_NORMAL_HIGH: extra = f" High temperature ({st:.0f}°C) is accelerating moisture loss."
            elif sm < 15: extra = " Moisture is critically low — crops may already be stressed."
        parts.append(f"⚠️ Irrigate within 24 hours — soil moisture is critically low.{extra}")
    if pest == "high":
        extra = ""
        if sensor and sensor.get("humidity", 0) > HUMIDITY_NORMAL_HIGH: extra = " High humidity is creating ideal conditions for pest spread."
        parts.append(f"🐛 High pest risk detected.{extra} Apply preventive treatment and inspect crops closely.")
    elif pest == "medium": parts.append("🔎 Moderate pest risk. Monitor field edges — conditions favour pest activity.")
    if planting == "optimal" and not irr and pest != "high": parts.append("🌱 Excellent conditions for planting. Soil and weather are ideal — plant now.")
    elif planting == "suboptimal" and not irr and pest == "low": parts.append("📊 Conditions acceptable but not ideal for planting. Wait 1–2 days for improvement.")
    if not parts: parts.append("✅ All indicators are safe. No immediate action required. Monitor daily.")
    return " ".join(parts)

# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    sms_mode = "not_configured"
    if AT_USERNAME and AT_API_KEY: sms_mode = "sandbox" if AT_USERNAME.lower() == "sandbox" else "live"
    admin_settings = firebase_get("soil_monitoring_system/admin_settings")
    sms_enabled = admin_settings.get("smsAlerts", True) if admin_settings else True
    elapsed = time.time() - _server_start_time
    grace_remaining = max(0, int(STARTUP_GRACE_SECONDS - elapsed))
    return {
        "status": "ok", "version": "9.5.0",
        "models_loaded": list(_state["models"].keys()),
        "is_training": _state["meta"]["is_training"],
        "sms_mode": sms_mode,
        "sms_admin_enabled": sms_enabled,
        "at_username": AT_USERNAME or "NOT SET",
        "pest_profiles": len(VIBRATION_PROFILES),
        "startup_grace_s": STARTUP_GRACE_SECONDS,
        "grace_remaining_s": grace_remaining,
        "sms_cooldown_s": SMS_COOLDOWN,
    }

@app.get("/model-status")
def model_status():
    return _state["meta"]

@app.get("/admin-sms-status")
def admin_sms_status():
    admin_settings = firebase_get("soil_monitoring_system/admin_settings")
    sms_enabled = admin_settings.get("smsAlerts", True) if admin_settings else True
    return {"smsAlerts": sms_enabled, "autoRefresh": admin_settings.get("autoRefresh", True) if admin_settings else True, "maintenanceMode": admin_settings.get("maintenanceMode", False) if admin_settings else False}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: SensorInput, background_tasks: BackgroundTasks):
    if not _state["models"] or _state["scaler"] is None:
        raise HTTPException(status_code=503, detail="Models are still initialising.")
    X = np.array([[data.soil_moisture, data.soil_temperature, data.pest_presence, data.air_temperature, data.rainfall, data.humidity, data.wind_speed]])
    Xs = _state["scaler"].transform(X)
    irr_p = _state["models"]["irrigation_needed"].predict_proba(Xs)[0]
    irr_c = int(_state["models"]["irrigation_needed"].predict(Xs)[0])
    pest_p = _state["models"]["pest_risk"].predict_proba(Xs)[0]
    pest_c = int(_state["models"]["pest_risk"].predict(Xs)[0])
    plan_p = _state["models"]["planting_window"].predict_proba(Xs)[0]
    plan_c = int(_state["models"]["planting_window"].predict(Xs)[0])
    pest_label = PEST_MAP[pest_c]
    planting_label = PLANTING_MAP[plan_c]
    importance = dict(zip(FEATURES, _state["models"]["irrigation_needed"].feature_importances_.round(3).tolist()))
    sensor_dict = data.model_dump()
    recommendation = build_recommendation(bool(irr_c), pest_label, planting_label, sensor_dict)
    if _conditions_warrant_sms(bool(irr_c), pest_label, planting_label, sensor_dict):
        firebase_set("soil_monitoring_system/aiRecommendation", {"text": recommendation, "timestamp": datetime.utcnow().isoformat() + "Z", "irrigation_needed": bool(irr_c), "pest_risk": pest_label, "planting_window": planting_label})
        background_tasks.add_task(auto_send_sms_to_farmers)
    return PredictionResponse(irrigationNeeded=bool(irr_c), irrigationConfidence=float(irr_p[irr_c]), pestRisk=pest_label, pestRiskScore=float(pest_p[pest_c]), plantingWindow=planting_label, plantingScore=float(plan_p[plan_c]), recommendation=recommendation, featureImportance=importance, modelMeta=_state["meta"])

@app.post("/detect_pest", response_model=PestDetectResponse)
def detect_pest(data: PestDetectInput):
    signal_arr = np.array(data.signal, dtype=float)
    try: results = identify_pest_signal(raw_signal=signal_arr, sampling_rate=data.sampling_rate, top_n=data.top_n, min_duration_sec=data.min_duration_sec)
    except ValueError as e: raise HTTPException(status_code=400, detail=str(e))
    report = generate_vibration_report(results, sensor_depth_cm=data.sensor_depth_cm)
    matches = [PestMatchResult(rank=r.rank, pest=r.pest.common_name, scientific_name=r.pest.name.replace("_", " "), score=r.score, confidence=r.confidence, peak_freq_hz=r.peak_freq_hz, burst_rate_per_sec=r.burst_rate_per_sec, amplitude_db=r.amplitude_db, temporal_pattern=r.temporal_pattern, damage_type=r.pest.damage_type, treatment=r.pest.treatment) for r in results]
    if not matches: summary = "No significant insect vibration detected."
    elif matches[0].confidence == "High": summary = f"Strong match: {matches[0].pest} detected."
    elif matches[0].confidence == "Medium": summary = f"Possible activity: {matches[0].pest}."
    else: summary = f"Weak signal — possible {matches[0].pest}."
    return PestDetectResponse(matches=matches, summary=summary, report=report, sensor_depth_cm=data.sensor_depth_cm, signal_duration_sec=round(len(signal_arr)/data.sampling_rate, 2))

@app.post("/sync-and-retrain")
def force_retrain(background_tasks: BackgroundTasks):
    if _retrain_lock.locked(): return {"status": "already_running"}
    background_tasks.add_task(sync_and_train)
    return {"status": "started"}

@app.get("/pest-profiles")
def pest_profiles():
    return [{"name": p.common_name, "scientific_name": p.name.replace("_", " "), "freq_range_hz": f"{p.freq_min}–{p.freq_max}", "treatment": p.treatment} for p in VIBRATION_PROFILES]

@app.post("/send-otp")
async def send_otp(body: OtpRequest):
    if not AT_USERNAME or not AT_API_KEY: return JSONResponse({"success": False, "error": "SMS not configured."}, status_code=503)
    try:
        phone = clean_phone(body.phone)
        otp = str(random.randint(100000, 999999))
        otp_store[phone] = {"otp": otp, "expires": time.time() + 600}
        message = f"AGRI ALERT: Your SoilApp verification code is {otp}. Valid for 10 minutes."
        response = sms_client.send(message, [phone])
        return JSONResponse({"success": True, "message": f"OTP sent to {phone}"})
    except Exception as e: return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/verify-otp")
async def verify_otp(body: OtpVerifyRequest):
    try:
        phone = clean_phone(body.phone)
        record = otp_store.get(phone)
        if not record: return JSONResponse({"success": False, "error": "No OTP found."}, status_code=400)
        if time.time() > record["expires"]: del otp_store[phone]; return JSONResponse({"success": False, "error": "OTP expired."}, status_code=400)
        if record["otp"] != body.otp.strip(): return JSONResponse({"success": False, "error": "Incorrect OTP."}, status_code=400)
        del otp_store[phone]
        return JSONResponse({"success": True, "message": "Phone verified."})
    except Exception as e: return JSONResponse({"success": False, "error": str(e)}, status_code=500)

@app.post("/send-sms-recommendation")
async def send_sms_recommendation(body: SmsRequest):
    """
    Manual SMS trigger — used by admin from the app.
    Always sends regardless of admin settings (admin override).
    """
    if not AT_USERNAME or not AT_API_KEY:
        return JSONResponse({"success": False, "error": "SMS credentials not configured."}, status_code=503)
    if not body.farmers or not body.recommendation.strip():
        return JSONResponse({"success": False, "error": "Farmers list and recommendation required"}, status_code=400)

    alert_type = _derive_alert_type(body.recommendation)
    results, now = [], time.time()

    for farmer in body.farmers:
        phone = clean_phone(farmer.phone)
        name = farmer.name or "Farmer"
        if not phone: results.append({"phone": "unknown", "status": "skipped — no phone"}); continue
        if _is_sms_on_cooldown(phone, alert_type, now):
            results.append({"phone": phone, "status": f"skipped — cooldown"})
            continue
        try:
            message = f"AGRI ALERT for {name}:\n{body.recommendation}\n— SoilApp {datetime.now().strftime('%d/%m/%Y %H:%M')}"
            if len(message) > 320: message = message[:317] + "..."
            response = sms_client.send(message, [phone])
            _mark_sms_sent(phone, alert_type, now)
            results.append({"phone": phone, "status": "sent"})
            print(f"[SMS] ✅ Admin sent to {phone} ({name}) | {response}")
        except Exception as e:
            results.append({"phone": phone, "status": f"error: {str(e)}"})

    return JSONResponse({"success": True, "sent": sum(1 for r in results if r["status"] == "sent"), "skipped": sum(1 for r in results if "skipped" in r["status"]), "errors": sum(1 for r in results if "error" in r["status"]), "results": results})

@app.get("/normal-ranges")
def normal_ranges():
    return {"soil_moisture_%": {"normal": f"{MOISTURE_NORMAL_LOW}–{MOISTURE_NORMAL_HIGH}"}, "sms_cooldown_seconds": SMS_COOLDOWN, "startup_grace_seconds": STARTUP_GRACE_SECONDS}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)