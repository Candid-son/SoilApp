/**
 * AI Analysis Dashboard — app/admin/ai-analysis.tsx (v6)
 *
 * FIXED in v6:
 *  - BACKEND_URL updated from placeholder → https://soil-pest-api.onrender.com
 */

import { useRouter } from "expo-router";
import { onValue, ref } from "firebase/database";
import { useCallback, useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Animated,
  Easing,
  ImageBackground,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { database } from "../../firebase/firebaseConfig";

// ─── Types ────────────────────────────────────────────────────────────────────

interface SensorData {
  soilMoisture:    number;
  soilTemperature: number;
  pestPresence:    number;
}

interface WeatherData {
  temperature: number;
  rainfall:    number;
  humidity:    number;
  windspeed:   number;
}

interface Prediction {
  irrigationNeeded:     boolean;
  irrigationConfidence: number;
  pestRisk:             "low" | "medium" | "high";
  pestRiskScore:        number;
  plantingWindow:       "optimal" | "suboptimal" | "avoid";
  plantingScore:        number;
  recommendation:       string;
  featureImportance:    Record<string, number>;
  modelMeta:            ModelMeta;
}

interface ModelMeta {
  last_trained:       string | null;
  seed_samples:       number;
  history_samples:    number;
  total_samples:      number;
  is_training:        boolean;
  firebase_connected: boolean;
  history_path:       string;
  accuracy: {
    irrigation_needed: number | null;
    pest_risk:         number | null;
    planting_window:   number | null;
  };
}

// ─── Constants ────────────────────────────────────────────────────────────────

const LESOTHO_LAT  = -29.3167;
const LESOTHO_LON  =  27.4833;

// FIX v6: Correct Render URL (was placeholder "your-render-app")
const BACKEND_URL  = "https://soil-pest-api.onrender.com";

const FETCH_TIMEOUT_MS = 10000; // increased to 10s — Render free tier can be slow to wake

const FIREBASE_SENSOR_PATH = "soil_monitoring_system/sensors/device001";

const DEFAULT_SENSOR: SensorData = {
  soilMoisture:    0,
  soilTemperature: 0,
  pestPresence:    0,
};

const FEATURE_LABELS: Record<string, string> = {
  soil_moisture:    "Soil Moisture",
  soil_temperature: "Soil Temp",
  pest_presence:    "Pest Signal",
  air_temperature:  "Air Temp",
  rainfall:         "Rainfall",
  humidity:         "Humidity",
  wind_speed:       "Wind Speed",
};

// ─── Pest status string → 0 / 1 ──────────────────────────────────────────────

function parsePestStatus(value: string | number | undefined | null): number {
  if (value === null || value === undefined) return 0;
  if (typeof value === "number") return value > 0 ? 1 : 0;
  const cleaned = String(value).trim().toLowerCase();
  if (
    cleaned === "no pest activity" ||
    cleaned === "no pest"          ||
    cleaned === "clear"            ||
    cleaned === "none"             ||
    cleaned === "normal"           ||
    cleaned === "no pest detected"
  ) return 0;
  return 1;
}

// ─── Fetch with timeout ───────────────────────────────────────────────────────

async function fetchWithTimeout(
  url: string,
  options: RequestInit = {},
  ms: number = FETCH_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), ms);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timer);
  }
}

// ─── Open-Meteo ───────────────────────────────────────────────────────────────

async function fetchLesothoWeather(): Promise<WeatherData> {
  try {
    const url =
      `https://api.open-meteo.com/v1/forecast` +
      `?latitude=${LESOTHO_LAT}&longitude=${LESOTHO_LON}` +
      `&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m` +
      `&timezone=Africa%2FJohannesburg`;
    const res  = await fetchWithTimeout(url, {}, 8000);
    const json = await res.json();
    const c    = json.current;
    return {
      temperature: c.temperature_2m,
      rainfall:    c.precipitation,
      humidity:    c.relative_humidity_2m,
      windspeed:   c.wind_speed_10m,
    };
  } catch {
    return { temperature: 18, rainfall: 0, humidity: 55, windspeed: 12 };
  }
}

// ─── Backend helpers ──────────────────────────────────────────────────────────

async function fetchPrediction(
  sensor: SensorData,
  weather: WeatherData,
): Promise<Prediction> {
  try {
    const res = await fetchWithTimeout(`${BACKEND_URL}/predict`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        soil_moisture:    sensor.soilMoisture,
        soil_temperature: sensor.soilTemperature,
        pest_presence:    sensor.pestPresence,
        air_temperature:  weather.temperature,
        rainfall:         weather.rainfall,
        humidity:         weather.humidity,
        wind_speed:       weather.windspeed,
      }),
    });
    return await res.json();
  } catch {
    // Rule-based fallback when backend is unreachable / waking up
    const irrigationNeeded = sensor.soilMoisture < 35;
    const pestScore =
      sensor.pestPresence * 0.6 +
      (weather.humidity > 75 ? 0.25 : 0) +
      (sensor.soilTemperature > 28 ? 0.15 : 0);
    const pestRisk: "low" | "medium" | "high" =
      pestScore > 0.6 ? "high" : pestScore > 0.3 ? "medium" : "low";
    const plantingScore =
      sensor.soilMoisture > 40 && sensor.soilMoisture < 70 &&
      sensor.soilTemperature > 15 && sensor.soilTemperature < 25
        ? 0.85
        : sensor.soilMoisture < 20
        ? 0.2
        : 0.5;
    const plantingWindow: "optimal" | "suboptimal" | "avoid" =
      plantingScore > 0.7 ? "optimal" : plantingScore > 0.4 ? "suboptimal" : "avoid";

    return {
      irrigationNeeded,
      irrigationConfidence: 0.75,
      pestRisk,
      pestRiskScore:  pestScore,
      plantingWindow,
      plantingScore,
      recommendation: irrigationNeeded
        ? "Soil moisture is critically low. Irrigate within 24 hours."
        : sensor.soilMoisture === 0
        ? "No sensor data — estimates based on defaults. Check IoT device connection."
        : "Backend offline or waking up — showing rule-based estimate. Try refreshing in 30s.",
      featureImportance: {},
      modelMeta: {
        last_trained: null, seed_samples: 0, history_samples: 0,
        total_samples: 0,   is_training: false, firebase_connected: false,
        history_path: "soil_monitoring_system/History",
        accuracy: { irrigation_needed: null, pest_risk: null, planting_window: null },
      },
    };
  }
}

async function fetchModelStatus(): Promise<ModelMeta | null> {
  try {
    const res = await fetchWithTimeout(`${BACKEND_URL}/model-status`);
    return await res.json();
  } catch {
    return null;
  }
}

async function triggerSyncRetrain(): Promise<string> {
  try {
    const res  = await fetchWithTimeout(
      `${BACKEND_URL}/sync-and-retrain`,
      { method: "POST" },
      30000, // retrain can take a while — allow 30s
    );
    const json = await res.json();
    return json.message || json.status;
  } catch {
    return "Backend unreachable or timed out — retrain skipped.";
  }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function GaugeBar({
  value, color, label, unit = "%",
}: {
  value: number; color: string; label: string; unit?: string;
}) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(anim, {
      toValue:         value / 100,
      duration:        900,
      easing:          Easing.out(Easing.cubic),
      useNativeDriver: false,
    }).start();
  }, [value]);
  const width = anim.interpolate({ inputRange: [0, 1], outputRange: ["0%", "100%"] });
  return (
    <View style={gb.wrap}>
      <View style={gb.labelRow}>
        <Text style={gb.label}>{label}</Text>
        <Text style={[gb.value, { color }]}>{value.toFixed(1)}{unit}</Text>
      </View>
      <View style={gb.track}>
        <Animated.View style={[gb.fill, { width, backgroundColor: color }]} />
      </View>
    </View>
  );
}
const gb = StyleSheet.create({
  wrap:     { marginBottom: 14 },
  labelRow: { flexDirection: "row", justifyContent: "space-between", marginBottom: 6 },
  label:    { fontSize: 12, color: "rgba(255,255,255,0.65)", letterSpacing: 0.5, textTransform: "uppercase", fontWeight: "600" },
  value:    { fontSize: 13, fontWeight: "800" },
  track:    { height: 6, backgroundColor: "rgba(255,255,255,0.1)", borderRadius: 3, overflow: "hidden" },
  fill:     { height: 6, borderRadius: 3 },
});

function ConfidenceBar({ value, color }: { value: number; color: string }) {
  const anim = useRef(new Animated.Value(0)).current;
  useEffect(() => {
    Animated.timing(anim, {
      toValue: value, duration: 800, easing: Easing.out(Easing.cubic), useNativeDriver: false,
    }).start();
  }, [value]);
  const width = anim.interpolate({ inputRange: [0, 1], outputRange: ["0%", "100%"] });
  return (
    <View style={cb.track}>
      <Animated.View style={[cb.fill, { width, backgroundColor: color }]} />
    </View>
  );
}
const cb = StyleSheet.create({
  track: { height: 4, backgroundColor: "rgba(255,255,255,0.1)", borderRadius: 2, marginTop: 10, overflow: "hidden" },
  fill:  { height: 4, borderRadius: 2 },
});

function RiskBadge({ level }: { level: "low" | "medium" | "high" }) {
  const map = {
    low:    { bg: "#1b5e2022", border: "#4caf50", text: "#4caf50", label: "LOW RISK" },
    medium: { bg: "#e6510022", border: "#ff9800", text: "#ff9800", label: "MED RISK" },
    high:   { bg: "#b71c1c22", border: "#f44336", text: "#f44336", label: "HIGH RISK" },
  };
  const s = map[level];
  return (
    <View style={[rb.badge, { backgroundColor: s.bg, borderColor: s.border }]}>
      <Text style={[rb.text, { color: s.text }]}>{s.label}</Text>
    </View>
  );
}

function PlantingBadge({ window: w }: { window: "optimal" | "suboptimal" | "avoid" }) {
  const map = {
    optimal:    { bg: "#1b5e2022", border: "#66bb6a", text: "#66bb6a", label: "✓ OPTIMAL" },
    suboptimal: { bg: "#f57f1722", border: "#ffa726", text: "#ffa726", label: "~ SUBOPTIMAL" },
    avoid:      { bg: "#b71c1c22", border: "#ef5350", text: "#ef5350", label: "✕ AVOID" },
  };
  const s = map[w];
  return (
    <View style={[rb.badge, { backgroundColor: s.bg, borderColor: s.border }]}>
      <Text style={[rb.text, { color: s.text }]}>{s.label}</Text>
    </View>
  );
}
const rb = StyleSheet.create({
  badge: { borderRadius: 8, borderWidth: 1, paddingHorizontal: 10, paddingVertical: 4 },
  text:  { fontSize: 11, fontWeight: "800", letterSpacing: 1 },
});

function WeatherTile({ icon, label, value }: { icon: string; label: string; value: string }) {
  return (
    <View style={wt.tile}>
      <Text style={wt.icon}>{icon}</Text>
      <Text style={wt.value}>{value}</Text>
      <Text style={wt.label}>{label}</Text>
    </View>
  );
}
const wt = StyleSheet.create({
  tile:  { width: "22%", alignItems: "center", backgroundColor: "rgba(255,255,255,0.08)", borderRadius: 12, padding: 10 },
  icon:  { fontSize: 18, marginBottom: 4 },
  value: { fontSize: 13, fontWeight: "800", color: "#fff" },
  label: { fontSize: 10, color: "rgba(255,255,255,0.55)", marginTop: 2, textTransform: "uppercase", letterSpacing: 0.3 },
});

function NoSensorBanner() {
  return (
    <View style={nsb.wrap}>
      <Text style={nsb.icon}>⚠️</Text>
      <View style={{ flex: 1 }}>
        <Text style={nsb.title}>No Recent Sensor Readings</Text>
        <Text style={nsb.body}>
          Predictions below are estimates only. Ensure your ESP32 is running and writing to{" "}
          <Text style={nsb.path}>soil_monitoring_system/sensors/device001</Text> in Firebase.
        </Text>
      </View>
    </View>
  );
}
const nsb = StyleSheet.create({
  wrap:  { flexDirection: "row", gap: 10, backgroundColor: "rgba(255,152,0,0.12)", borderRadius: 12, padding: 14, marginBottom: 16, borderWidth: 1, borderColor: "rgba(255,152,0,0.35)", alignItems: "flex-start" },
  icon:  { fontSize: 18, marginTop: 1 },
  title: { fontSize: 13, fontWeight: "800", color: "#ffa726", marginBottom: 4 },
  body:  { fontSize: 12, color: "rgba(255,200,100,0.75)", lineHeight: 18 },
  path:  { fontWeight: "700", color: "#ffd54f" },
});

function ErrorState({ onRetry }: { onRetry: () => void }) {
  return (
    <View style={es.wrap}>
      <Text style={es.emoji}>🌐</Text>
      <Text style={es.title}>Could not load analysis</Text>
      <Text style={es.body}>
        Check your internet connection. The backend may be waking up (Render free tier sleeps after inactivity — wait 30s and retry).
      </Text>
      <TouchableOpacity style={es.btn} onPress={onRetry}>
        <Text style={es.btnText}>Try Again</Text>
      </TouchableOpacity>
    </View>
  );
}
const es = StyleSheet.create({
  wrap:    { flex: 1, justifyContent: "center", alignItems: "center", padding: 32, gap: 12 },
  emoji:   { fontSize: 48 },
  title:   { fontSize: 16, fontWeight: "800", color: "#fff", textAlign: "center" },
  body:    { fontSize: 13, color: "rgba(255,255,255,0.5)", textAlign: "center", lineHeight: 20 },
  btn:     { marginTop: 8, backgroundColor: "#2e7d32", borderRadius: 12, paddingHorizontal: 24, paddingVertical: 12 },
  btnText: { color: "#fff", fontWeight: "700", fontSize: 14 },
});

function LearningStatusPanel({
  meta, onSyncRetrain, syncing, syncMessage,
}: {
  meta: ModelMeta; onSyncRetrain: () => void; syncing: boolean; syncMessage: string;
}) {
  const pct = meta.total_samples > 0
    ? Math.round((meta.history_samples / meta.total_samples) * 100)
    : 0;

  const fmtDate = (iso: string | null) => {
    if (!iso) return "Not yet trained";
    try { return new Date(iso).toLocaleString(); } catch { return iso; }
  };

  const fmtAcc = (v: number | null) =>
    v !== null ? `${(v * 100).toFixed(1)}%` : "—";

  return (
    <View style={lp.card}>
      <View style={lp.header}>
        <View style={lp.headerLeft}>
          <Text style={lp.title}>🧠  AI Learning Status</Text>
          <View style={[lp.dot, { backgroundColor: meta.firebase_connected ? "#4caf50" : "#f44336" }]} />
          <Text style={[lp.dotLabel, { color: meta.firebase_connected ? "#4caf50" : "#f44336" }]}>
            {meta.firebase_connected ? "Firebase connected" : "Firebase offline"}
          </Text>
        </View>
        {meta.is_training && (
          <View style={lp.trainBadge}>
            <ActivityIndicator size="small" color="#ff9800" />
            <Text style={lp.trainText}>Training…</Text>
          </View>
        )}
      </View>

      <View style={lp.statsRow}>
        {[
          { num: meta.history_samples, label: "Firebase rows" },
          { num: meta.seed_samples,    label: "Seed data" },
          { num: meta.total_samples,   label: "Total trained" },
        ].map(({ num, label }) => (
          <View key={label} style={lp.stat}>
            <Text style={lp.statNum}>{num}</Text>
            <Text style={lp.statLabel}>{label}</Text>
          </View>
        ))}
      </View>

      <View style={lp.ratioWrap}>
        <View style={lp.ratioRow}>
          <Text style={lp.ratioLabel}>Real data ratio</Text>
          <Text style={lp.ratioPct}>{pct}%</Text>
        </View>
        <View style={lp.ratioTrack}>
          <View style={[lp.ratioFill, { width: `${pct}%` }]} />
        </View>
        <Text style={lp.ratioNote}>
          {pct < 20
            ? "Still learning — model leans on seed data. Keep collecting readings."
            : pct < 60
            ? "Good progress — Firebase history is influencing predictions."
            : "High confidence — model is mostly trained on your real field data."}
        </Text>
      </View>

      <View style={lp.accTable}>
        <Text style={lp.accTitle}>Model Accuracy</Text>
        {[
          { label: "Irrigation", key: "irrigation_needed" as const },
          { label: "Pest Risk",  key: "pest_risk"         as const },
          { label: "Planting",   key: "planting_window"   as const },
        ].map(({ label, key }) => (
          <View key={key} style={lp.accRow}>
            <Text style={lp.accLabel}>{label}</Text>
            <Text style={lp.accVal}>{fmtAcc(meta.accuracy[key])}</Text>
          </View>
        ))}
      </View>

      <Text style={lp.lastTrained}>Last trained: {fmtDate(meta.last_trained)}</Text>
      <Text style={lp.historyPath}>History path: /{meta.history_path}</Text>

      <TouchableOpacity
        style={[lp.syncBtn, (syncing || meta.is_training) && lp.syncBtnDisabled]}
        onPress={onSyncRetrain}
        disabled={syncing || meta.is_training}
        activeOpacity={0.8}
      >
        {syncing
          ? <ActivityIndicator size="small" color="#fff" />
          : <Text style={lp.syncBtnText}>🔄  Sync Firebase & Retrain</Text>
        }
      </TouchableOpacity>

      {!!syncMessage && <Text style={lp.syncMsg}>{syncMessage}</Text>}
    </View>
  );
}

const lp = StyleSheet.create({
  card:       { backgroundColor: "rgba(255,255,255,0.07)", borderRadius: 16, padding: 18, marginBottom: 16, borderWidth: 1, borderColor: "rgba(255,255,255,0.12)" },
  header:     { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 14 },
  headerLeft: { flexDirection: "row", alignItems: "center", gap: 8, flexWrap: "wrap" },
  title:      { fontSize: 13, fontWeight: "800", color: "#fff" },
  dot:        { width: 8, height: 8, borderRadius: 4 },
  dotLabel:   { fontSize: 11, fontWeight: "600" },
  trainBadge: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: "rgba(255,152,0,0.15)", borderRadius: 8, paddingHorizontal: 8, paddingVertical: 4 },
  trainText:  { fontSize: 11, color: "#ff9800", fontWeight: "700" },
  statsRow:   { flexDirection: "row", justifyContent: "space-around", marginBottom: 16 },
  stat:       { alignItems: "center" },
  statNum:    { fontSize: 22, fontWeight: "800", color: "#4dd0e1" },
  statLabel:  { fontSize: 10, color: "rgba(255,255,255,0.5)", marginTop: 2, textTransform: "uppercase", letterSpacing: 0.3 },
  ratioWrap:  { marginBottom: 14 },
  ratioRow:   { flexDirection: "row", justifyContent: "space-between", marginBottom: 6 },
  ratioLabel: { fontSize: 11, color: "rgba(255,255,255,0.55)", textTransform: "uppercase", letterSpacing: 0.4 },
  ratioPct:   { fontSize: 11, color: "#4caf50", fontWeight: "800" },
  ratioTrack: { height: 8, backgroundColor: "rgba(255,255,255,0.1)", borderRadius: 4, overflow: "hidden", marginBottom: 6 },
  ratioFill:  { height: 8, backgroundColor: "#4caf50", borderRadius: 4 },
  ratioNote:  { fontSize: 11, color: "rgba(255,255,255,0.4)", lineHeight: 16 },
  accTable:   { backgroundColor: "rgba(0,0,0,0.2)", borderRadius: 10, padding: 12, marginBottom: 12 },
  accTitle:   { fontSize: 11, color: "rgba(255,255,255,0.45)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 8 },
  accRow:     { flexDirection: "row", justifyContent: "space-between", paddingVertical: 4, borderBottomWidth: 1, borderBottomColor: "rgba(255,255,255,0.06)" },
  accLabel:   { fontSize: 13, color: "rgba(255,255,255,0.7)" },
  accVal:     { fontSize: 13, fontWeight: "800", color: "#66bb6a" },
  lastTrained:{ fontSize: 11, color: "rgba(255,255,255,0.35)", marginBottom: 4 },
  historyPath:{ fontSize: 11, color: "rgba(255,255,255,0.25)", marginBottom: 14 },
  syncBtn:         { backgroundColor: "#00796b", borderRadius: 12, padding: 13, alignItems: "center" },
  syncBtnDisabled: { backgroundColor: "#004d40" },
  syncBtnText:     { color: "#fff", fontSize: 13, fontWeight: "800" },
  syncMsg:         { fontSize: 11, color: "#80cbc4", textAlign: "center", marginTop: 8 },
});

function FeatureImportanceChart({ importance }: { importance: Record<string, number> }) {
  const entries = Object.entries(importance).sort((a, b) => b[1] - a[1]).slice(0, 7);
  if (!entries.length) return null;
  const max = entries[0][1];
  return (
    <View style={fi.card}>
      <Text style={fi.title}>📊  Feature Importance</Text>
      <Text style={fi.sub}>Which sensor most influences irrigation predictions</Text>
      {entries.map(([key, val]) => (
        <View key={key} style={fi.row}>
          <Text style={fi.label}>{FEATURE_LABELS[key] ?? key}</Text>
          <View style={fi.track}>
            <View style={[fi.fill, { width: `${(val / max) * 100}%` }]} />
          </View>
          <Text style={fi.pct}>{(val * 100).toFixed(1)}%</Text>
        </View>
      ))}
    </View>
  );
}
const fi = StyleSheet.create({
  card:  { backgroundColor: "rgba(255,255,255,0.07)", borderRadius: 16, padding: 18, marginBottom: 16, borderWidth: 1, borderColor: "rgba(255,255,255,0.1)" },
  title: { fontSize: 13, fontWeight: "800", color: "#fff", marginBottom: 4 },
  sub:   { fontSize: 11, color: "rgba(255,255,255,0.4)", marginBottom: 14 },
  row:   { flexDirection: "row", alignItems: "center", marginBottom: 10, gap: 8 },
  label: { width: 90, fontSize: 11, color: "rgba(255,255,255,0.65)" },
  track: { flex: 1, height: 6, backgroundColor: "rgba(255,255,255,0.1)", borderRadius: 3, overflow: "hidden" },
  fill:  { height: 6, backgroundColor: "#4dd0e1", borderRadius: 3 },
  pct:   { width: 36, fontSize: 11, color: "#4dd0e1", fontWeight: "700", textAlign: "right" },
});

// ─── Main screen ──────────────────────────────────────────────────────────────

export default function AIAnalysisDashboard() {
  const router = useRouter();

  const [sensor,        setSensor]        = useState<SensorData | null>(null);
  const [sensorIsLive,  setSensorIsLive]  = useState(false);
  const [weather,       setWeather]       = useState<WeatherData | null>(null);
  const [prediction,    setPrediction]    = useState<Prediction | null>(null);
  const [modelMeta,     setModelMeta]     = useState<ModelMeta | null>(null);
  const [loading,       setLoading]       = useState(true);
  const [hasError,      setHasError]      = useState(false);
  const [refreshing,    setRefreshing]    = useState(false);
  const [syncing,       setSyncing]       = useState(false);
  const [syncMessage,   setSyncMessage]   = useState("");
  const [lastUpdated,   setLastUpdated]   = useState("");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  const fadeAnim         = useRef(new Animated.Value(0)).current;
  const hasRunRef        = useRef(false);
  const firebaseTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── Core analysis ──────────────────────────────────────────────────────────
  const runAnalysis = useCallback(async (sensorData: SensorData) => {
    setHasError(false);
    try {
      const [w, meta] = await Promise.all([
        fetchLesothoWeather(),
        fetchModelStatus(),
      ]);
      setWeather(w);
      setBackendOnline(meta !== null);
      if (meta) setModelMeta(meta);

      const p = await fetchPrediction(sensorData, w);
      setPrediction(p);
      if (p.modelMeta && !meta) setModelMeta(p.modelMeta);

      setLastUpdated(new Date().toLocaleTimeString());
      Animated.timing(fadeAnim, { toValue: 1, duration: 600, useNativeDriver: true }).start();
    } catch (e) {
      console.error("Analysis error:", e);
      setHasError(true);
    } finally {
      setLoading(false);
      setRefreshing(false);
      hasRunRef.current = true;
    }
  }, [fadeAnim]);

  // ── Firebase listener ──────────────────────────────────────────────────────
  useEffect(() => {
    firebaseTimerRef.current = setTimeout(() => {
      if (!hasRunRef.current) {
        runAnalysis(DEFAULT_SENSOR);
      }
    }, 5000);

    const sensorRef = ref(database, FIREBASE_SENSOR_PATH);

    const unsubscribe = onValue(sensorRef, (snap) => {
      const d = snap.val();
      if (d) {
        if (firebaseTimerRef.current) {
          clearTimeout(firebaseTimerRef.current);
          firebaseTimerRef.current = null;
        }

        const liveSensor: SensorData = {
          soilMoisture:    d.moisture    ?? 0,
          soilTemperature: d.temperature ?? 0,
          pestPresence:    parsePestStatus(d.pest_status),
        };

        setSensor(liveSensor);
        setSensorIsLive(true);
        runAnalysis(liveSensor);
      }
    });

    return () => {
      unsubscribe();
      if (firebaseTimerRef.current) clearTimeout(firebaseTimerRef.current);
    };
  }, [runAnalysis]);

  // ── Refresh ────────────────────────────────────────────────────────────────
  const handleRefresh = () => {
    setRefreshing(true);
    fadeAnim.setValue(0);
    runAnalysis(sensor ?? DEFAULT_SENSOR);
  };

  // ── Sync & retrain ─────────────────────────────────────────────────────────
  const handleSyncRetrain = async () => {
    setSyncing(true);
    setSyncMessage("");
    const msg = await triggerSyncRetrain();
    setSyncMessage(msg);
    setSyncing(false);
    setTimeout(async () => {
      const meta = await fetchModelStatus();
      if (meta) setModelMeta(meta);
    }, 3000);
  };

  const displaySensor = sensor ?? DEFAULT_SENSOR;

  return (
    <ImageBackground
      source={{ uri: "https://images.unsplash.com/photo-1563514227147-6d2ff665a6a0" }}
      style={styles.background}
    >
      <View style={styles.overlay}>

        {/* Top bar */}
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.backBtn} onPress={() => router.replace("/admin/dashboard")}>
            <Text style={styles.backArrow}>←</Text>
            <Text style={styles.backLabel}>Dashboard</Text>
          </TouchableOpacity>

          <View style={styles.topCenter}>
            <Text style={styles.screenTitle}>AI Analysis</Text>
            {backendOnline !== null && (
              <View style={[styles.statusChip, { borderColor: backendOnline ? "#4caf5055" : "#f4433655" }]}>
                <View style={[styles.statusDot, { backgroundColor: backendOnline ? "#4caf50" : "#f44336" }]} />
                <Text style={[styles.statusLabel, { color: backendOnline ? "#81c784" : "#e57373" }]}>
                  {backendOnline ? "ML online" : "Rule-based"}
                </Text>
              </View>
            )}
          </View>

          <TouchableOpacity style={styles.refreshBtn} onPress={handleRefresh} disabled={loading}>
            <Text style={styles.refreshIcon}>{refreshing ? "⏳" : "🔄"}</Text>
          </TouchableOpacity>
        </View>

        {loading ? (
          <View style={styles.loaderWrap}>
            <ActivityIndicator size="large" color="#4caf50" />
            <Text style={styles.loaderText}>
              Fetching Lesotho weather & running analysis…
            </Text>
            <Text style={styles.loaderSub}>
              Loading with defaults if no sensor data found (up to 5s)
            </Text>
          </View>
        ) : hasError ? (
          <ErrorState onRetry={handleRefresh} />
        ) : (
          <Animated.ScrollView
            style={{ opacity: fadeAnim }}
            contentContainerStyle={styles.scroll}
            showsVerticalScrollIndicator={false}
          >
            {/* Location chip */}
            <View style={styles.locationChip}>
              <Text style={styles.locationDot}>📍</Text>
              <Text style={styles.locationText}>Maseru, Lesotho</Text>
              {lastUpdated ? <Text style={styles.lastUpdated}>Updated {lastUpdated}</Text> : null}
            </View>

            {/* No-sensor warning */}
            {!sensorIsLive && <NoSensorBanner />}

            {/* AI Learning Status */}
            {modelMeta && (
              <LearningStatusPanel
                meta={modelMeta}
                onSyncRetrain={handleSyncRetrain}
                syncing={syncing}
                syncMessage={syncMessage}
              />
            )}

            {/* Weather */}
            {weather && (
              <View style={styles.card}>
                <Text style={styles.cardTitle}>🌤  Lesotho Weather  ·  Open-Meteo</Text>
                <View style={styles.weatherGrid}>
                  <WeatherTile icon="🌡" label="Air Temp" value={`${weather.temperature}°C`} />
                  <WeatherTile icon="💧" label="Humidity" value={`${weather.humidity}%`}     />
                  <WeatherTile icon="🌧" label="Rainfall" value={`${weather.rainfall}mm`}    />
                  <WeatherTile icon="💨" label="Wind"     value={`${weather.windspeed}km/h`} />
                </View>
              </View>
            )}

            {/* Sensor readings */}
            <View style={styles.card}>
              <View style={styles.cardTitleRow}>
                <Text style={styles.cardTitle}>📡  Sensor Readings</Text>
                <View style={[
                  styles.sensorSourceBadge,
                  { backgroundColor: sensorIsLive ? "rgba(76,175,80,0.15)" : "rgba(255,152,0,0.15)",
                    borderColor:      sensorIsLive ? "rgba(76,175,80,0.4)"  : "rgba(255,152,0,0.4)" },
                ]}>
                  <Text style={[styles.sensorSourceText, { color: sensorIsLive ? "#81c784" : "#ffb74d" }]}>
                    {sensorIsLive ? "● LIVE" : "○ DEFAULT"}
                  </Text>
                </View>
              </View>

              <GaugeBar value={displaySensor.soilMoisture} color="#2196f3" label="Soil Moisture" />
              <GaugeBar
                value={(displaySensor.soilTemperature / 50) * 100}
                color="#ff9800"
                label="Soil Temperature"
                unit={`°C (${displaySensor.soilTemperature})`}
              />
              <View style={styles.pestRow}>
                <Text style={styles.pestLabel}>Pest Presence</Text>
                <View style={[styles.pestIndicator, {
                  backgroundColor: displaySensor.pestPresence ? "#f44336" : "#4caf50",
                }]}>
                  <Text style={styles.pestStatus}>
                    {displaySensor.pestPresence ? "⚠ DETECTED" : "✓ CLEAR"}
                  </Text>
                </View>
              </View>
            </View>

            {/* Predictions */}
            {prediction && (
              <>
                <Text style={styles.sectionHeader}>🤖  Random Forest Predictions</Text>

                <View style={[styles.predCard, prediction.irrigationNeeded && styles.predCardAlert]}>
                  <View style={styles.predHeader}>
                    <Text style={styles.predIcon}>💧</Text>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.predTitle}>Irrigation Needed</Text>
                      <Text style={styles.predConf}>
                        Confidence: {(prediction.irrigationConfidence * 100).toFixed(0)}%
                      </Text>
                    </View>
                    <View style={[styles.boolBadge, {
                      backgroundColor: prediction.irrigationNeeded ? "#f44336" : "#4caf50",
                    }]}>
                      <Text style={styles.boolText}>
                        {prediction.irrigationNeeded ? "YES" : "NO"}
                      </Text>
                    </View>
                  </View>
                  <ConfidenceBar
                    value={prediction.irrigationConfidence}
                    color={prediction.irrigationNeeded ? "#f44336" : "#4caf50"}
                  />
                </View>

                <View style={styles.predCard}>
                  <View style={styles.predHeader}>
                    <Text style={styles.predIcon}>🐛</Text>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.predTitle}>Pest Risk Assessment</Text>
                      <Text style={styles.predConf}>
                        Score: {(prediction.pestRiskScore * 100).toFixed(0)}%
                      </Text>
                    </View>
                    <RiskBadge level={prediction.pestRisk} />
                  </View>
                  <ConfidenceBar
                    value={prediction.pestRiskScore}
                    color={
                      prediction.pestRisk === "high"   ? "#f44336" :
                      prediction.pestRisk === "medium" ? "#ff9800" : "#4caf50"
                    }
                  />
                </View>

                <View style={styles.predCard}>
                  <View style={styles.predHeader}>
                    <Text style={styles.predIcon}>🌱</Text>
                    <View style={{ flex: 1 }}>
                      <Text style={styles.predTitle}>Planting Window</Text>
                      <Text style={styles.predConf}>
                        Score: {(prediction.plantingScore * 100).toFixed(0)}%
                      </Text>
                    </View>
                    <PlantingBadge window={prediction.plantingWindow} />
                  </View>
                  <ConfidenceBar
                    value={prediction.plantingScore}
                    color={
                      prediction.plantingWindow === "optimal"    ? "#4caf50" :
                      prediction.plantingWindow === "suboptimal" ? "#ff9800" : "#f44336"
                    }
                  />
                </View>

                <View style={styles.recommendBox}>
                  <Text style={styles.recommendTitle}>💡 AI Recommendation</Text>
                  <Text style={styles.recommendText}>{prediction.recommendation}</Text>
                </View>

                {Object.keys(prediction.featureImportance).length > 0 && (
                  <FeatureImportanceChart importance={prediction.featureImportance} />
                )}
              </>
            )}

            <TouchableOpacity
              style={styles.backDashBtn}
              onPress={() => router.replace("/admin/dashboard")}
            >
              <Text style={styles.backDashText}>← Back to Admin Dashboard</Text>
            </TouchableOpacity>
            <View style={{ height: 32 }} />
          </Animated.ScrollView>
        )}
      </View>
    </ImageBackground>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const styles = StyleSheet.create({
  background: { flex: 1 },
  overlay:    { flex: 1, backgroundColor: "rgba(5,20,10,0.82)" },
  topBar: {
    flexDirection: "row", alignItems: "center", justifyContent: "space-between",
    paddingHorizontal: 20, paddingTop: 56, paddingBottom: 16,
    borderBottomWidth: 1, borderBottomColor: "rgba(255,255,255,0.08)",
  },
  backBtn:     { flexDirection: "row", alignItems: "center", gap: 6, minWidth: 80 },
  backArrow:   { fontSize: 20, color: "#4caf50" },
  backLabel:   { fontSize: 14, color: "#4caf50", fontWeight: "700" },
  topCenter:   { alignItems: "center", gap: 4 },
  screenTitle: { fontSize: 17, fontWeight: "800", color: "#fff", letterSpacing: 0.5 },
  statusChip:  { flexDirection: "row", alignItems: "center", gap: 5, borderWidth: 1, borderRadius: 10, paddingHorizontal: 8, paddingVertical: 3 },
  statusDot:   { width: 6, height: 6, borderRadius: 3 },
  statusLabel: { fontSize: 10, fontWeight: "700", letterSpacing: 0.3 },
  refreshBtn:  { padding: 4, minWidth: 40, alignItems: "flex-end" },
  refreshIcon: { fontSize: 18 },
  loaderWrap: { flex: 1, justifyContent: "center", alignItems: "center", gap: 12 },
  loaderText: { color: "rgba(255,255,255,0.7)", fontSize: 14, textAlign: "center", paddingHorizontal: 40, fontWeight: "600" },
  loaderSub:  { color: "rgba(255,255,255,0.35)", fontSize: 12, textAlign: "center", paddingHorizontal: 40 },
  scroll: { padding: 20, paddingTop: 16 },
  locationChip: {
    flexDirection: "row", alignItems: "center", gap: 6, marginBottom: 18,
    backgroundColor: "rgba(76,175,80,0.12)", borderRadius: 20,
    paddingHorizontal: 14, paddingVertical: 7, alignSelf: "flex-start",
    borderWidth: 1, borderColor: "rgba(76,175,80,0.3)",
  },
  locationDot:  { fontSize: 12 },
  locationText: { fontSize: 12, color: "#4caf50", fontWeight: "700" },
  lastUpdated:  { fontSize: 11, color: "rgba(255,255,255,0.4)", marginLeft: 8 },
  card: {
    backgroundColor: "rgba(255,255,255,0.07)", borderRadius: 16, padding: 18,
    marginBottom: 16, borderWidth: 1, borderColor: "rgba(255,255,255,0.1)",
  },
  cardTitleRow: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 14 },
  cardTitle:    { fontSize: 13, fontWeight: "800", color: "#fff", letterSpacing: 0.5 },
  weatherGrid:  { flexDirection: "row", justifyContent: "space-between", gap: 6 },
  sensorSourceBadge: { borderRadius: 8, borderWidth: 1, paddingHorizontal: 8, paddingVertical: 3 },
  sensorSourceText:  { fontSize: 10, fontWeight: "800", letterSpacing: 0.5 },
  pestRow:      { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginTop: 4 },
  pestLabel:    { fontSize: 12, color: "rgba(255,255,255,0.65)", textTransform: "uppercase", fontWeight: "600", letterSpacing: 0.5 },
  pestIndicator:{ borderRadius: 8, paddingHorizontal: 10, paddingVertical: 5 },
  pestStatus:   { fontSize: 11, fontWeight: "800", color: "#fff", letterSpacing: 0.5 },
  sectionHeader: {
    fontSize: 13, fontWeight: "800", color: "rgba(255,255,255,0.5)",
    letterSpacing: 1, textTransform: "uppercase", marginBottom: 12, marginTop: 4,
  },
  predCard:      { backgroundColor: "rgba(255,255,255,0.07)", borderRadius: 14, padding: 16, marginBottom: 12, borderWidth: 1, borderColor: "rgba(255,255,255,0.1)" },
  predCardAlert: { borderColor: "rgba(244,67,54,0.4)", backgroundColor: "rgba(244,67,54,0.08)" },
  predHeader:    { flexDirection: "row", alignItems: "center", gap: 12 },
  predIcon:      { fontSize: 24 },
  predTitle:     { fontSize: 14, fontWeight: "700", color: "#fff" },
  predConf:      { fontSize: 11, color: "rgba(255,255,255,0.45)", marginTop: 2 },
  boolBadge: { borderRadius: 8, paddingHorizontal: 12, paddingVertical: 5 },
  boolText:  { fontSize: 12, fontWeight: "800", color: "#fff", letterSpacing: 0.5 },
  recommendBox:   { backgroundColor: "rgba(76,175,80,0.12)", borderRadius: 14, padding: 18, marginBottom: 16, borderWidth: 1, borderColor: "rgba(76,175,80,0.3)" },
  recommendTitle: { fontSize: 13, fontWeight: "800", color: "#66bb6a", marginBottom: 8, letterSpacing: 0.5 },
  recommendText:  { fontSize: 14, color: "rgba(255,255,255,0.85)", lineHeight: 22 },
  backDashBtn:  { backgroundColor: "rgba(255,255,255,0.08)", borderRadius: 14, padding: 16, alignItems: "center", borderWidth: 1, borderColor: "rgba(255,255,255,0.15)" },
  backDashText: { color: "#4caf50", fontSize: 14, fontWeight: "700" },
});