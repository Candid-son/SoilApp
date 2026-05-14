import { useRouter } from "expo-router";
import { off, onValue, ref, set } from "firebase/database";
import React, { useEffect, useRef, useState } from "react";
import {
  Animated,
  Dimensions,
  Modal,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { LineChart } from "react-native-chart-kit";
import { database } from "../firebase/firebaseConfig";

const DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const RENDER_API_URL = "https://soilapp.onrender.com/detect_pest";

const getDayKey = (): string => {
  const now = new Date();
  const year = now.getFullYear();
  const startOfYear = new Date(year, 0, 1);
  const week = Math.ceil(
    ((now.getTime() - startOfYear.getTime()) / 86400000 + startOfYear.getDay() + 1) / 7
  );
  const dayAbbr = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][now.getDay()];
  return `${year}-W${String(week).padStart(2, "0")}-${dayAbbr}`;
};

const getWeekKey = (): string => getDayKey().split("-").slice(0, 2).join("-");

const getMonthKey = (): string => {
  const now = new Date();
  return `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
};

const avg = (arr: number[]): number =>
  arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;

const getPestColor = (level: number): string => {
  if (level < 0.2) return "#4caf50";
  if (level < 0.5) return "#ffb74d";
  return "#ef5350";
};

const getPestStatus = (level: number): string => {
  if (level === 0) return "No Pests Detected ✅";
  if (level < 0.5) return "Low Pest Activity ⚠️";
  return "Pests Detected 🚨";
};

const getPestRecommendation = (level: number): string => {
  if (level < 0.2) return "✅ No pest activity — fields are clear.\nContinue routine monitoring. All crops including maize, beans, potatoes, wheat, and sorghum are safe.";
  if (level < 0.5) return "⚠️ Low pest activity detected — monitor closely.\nInspect crops for early signs of infestation. Consider preventive measures for maize and beans.";
  return "🚨 Pest activity detected — take action immediately.\nApply appropriate pesticides. Maize, beans, and potatoes are most vulnerable. Inspect all fields urgently.";
};

type PestMatch = {
  pest: string;
  score: number;
  confidence: string;
  damage: string;
  treatment: string;
};

type WeekData = { weekKey: string; dailyAvgs: { [day: string]: number } };

const MonthlyAnalysis = ({
  visible, onClose, monthKey,
}: {
  visible: boolean; onClose: () => void; monthKey: string;
}) => {
  const [weeks, setWeeks] = useState<WeekData[]>([]);
  const screenWidth = Dimensions.get("window").width;

  useEffect(() => {
    if (!visible) return;
    const monthRef = ref(database, `soil_monitoring_system/sensors/device001/pestDailyAverages`);
    onValue(monthRef, (snapshot) => {
      const all: { [key: string]: number } = snapshot.val() || {};
      const [year] = monthKey.split("-");
      const relevant = Object.entries(all).filter(([key]) => key.split("-").length >= 3 && key.startsWith(year));
      const weekMap: { [wk: string]: { [day: string]: number } } = {};
      relevant.forEach(([key, value]) => {
        const parts = key.split("-");
        const wk = `${parts[0]}-${parts[1]}`;
        const day = parts[2];
        if (!weekMap[wk]) weekMap[wk] = {};
        weekMap[wk][day] = value;
      });
      const sorted: WeekData[] = Object.entries(weekMap).sort(([a], [b]) => a.localeCompare(b)).map(([wk, dailyAvgs]) => ({ weekKey: wk, dailyAvgs }));
      setWeeks(sorted);
    });
    return () => off(monthRef);
  }, [visible, monthKey]);

  const overallAvg = (): number => { const all = weeks.flatMap((w) => Object.values(w.dailyAvgs)); return all.length ? avg(all) : 0; };
  const highestRisk = (): { day: string; val: number } => { let best = { day: "—", val: -Infinity }; weeks.forEach((w) => Object.entries(w.dailyAvgs).forEach(([day, val]) => { if (val > best.val) best = { day, val }; })); return best; };
  const lowestRisk = (): { day: string; val: number } => { let best = { day: "—", val: Infinity }; weeks.forEach((w) => Object.entries(w.dailyAvgs).forEach(([day, val]) => { if (val < best.val) best = { day, val }; })); return best; };

  return (
    <Modal visible={visible} animationType="slide" transparent={false}>
      <View style={modal.container}>
        <View style={modal.header}>
          <TouchableOpacity onPress={onClose} style={modal.backBtn}><Text style={modal.backBtnText}>← Back</Text></TouchableOpacity>
          <Text style={modal.title}>📅 Pest Monthly Analysis</Text>
          <Text style={modal.subtitle}>{monthKey}</Text>
        </View>
        <ScrollView contentContainerStyle={modal.scroll}>
          <View style={modal.cardRow}>
            <View style={[modal.card, { borderColor: "#ffb74d" }]}>
              <Text style={modal.cardLabel}>Monthly Avg</Text>
              <Text style={[modal.cardValue, { color: getPestColor(overallAvg()) }]}>{overallAvg().toFixed(2)}</Text>
            </View>
            <View style={[modal.card, { borderColor: "#ef5350" }]}>
              <Text style={modal.cardLabel}>Highest Risk</Text>
              <Text style={[modal.cardValue, { color: "#ef5350" }]}>{highestRisk().val === -Infinity ? "—" : highestRisk().val.toFixed(2)}</Text>
              <Text style={modal.cardSub}>{highestRisk().day}</Text>
            </View>
            <View style={[modal.card, { borderColor: "#4caf50" }]}>
              <Text style={modal.cardLabel}>Lowest Risk</Text>
              <Text style={[modal.cardValue, { color: "#4caf50" }]}>{lowestRisk().val === Infinity ? "—" : lowestRisk().val.toFixed(2)}</Text>
              <Text style={modal.cardSub}>{lowestRisk().day}</Text>
            </View>
          </View>
          {weeks.length === 0 ? (
            <Text style={modal.empty}>No data available for this month yet.</Text>
          ) : (
            weeks.map((w, wi) => {
              const labels = DAYS.filter((d) => w.dailyAvgs[d] !== undefined);
              const values = labels.map((d) => w.dailyAvgs[d]);
              const weekAvg = avg(values);
              return (
                <View key={w.weekKey} style={modal.weekBlock}>
                  <View style={modal.weekHeader}>
                    <Text style={modal.weekTitle}>Week {wi + 1}</Text>
                    <Text style={[modal.weekAvg, { color: getPestColor(weekAvg) }]}>Avg: {weekAvg.toFixed(2)}</Text>
                  </View>
                  {labels.length > 1 && (
                    <View style={modal.chartWrapper}>
                      <View style={modal.yAxisLabelContainer}><Text style={modal.yAxisLabel}>Pest Level (0–1)</Text></View>
                      <View style={{ flex: 1 }}>
                        <LineChart
                          data={{ labels, datasets: [{ data: values, color: (o = 1) => `rgba(211,47,47,${o})`, strokeWidth: 2.5 }] }}
                          width={screenWidth - 88} height={180}
                          chartConfig={{ backgroundColor: "#111d2b", backgroundGradientFrom: "#111d2b", backgroundGradientTo: "#1b2d42", decimalPlaces: 1, color: (o = 1) => `rgba(255,255,255,${o})`, labelColor: (o = 1) => `rgba(180,200,220,${o})`, propsForDots: { r: "5", strokeWidth: "2", stroke: "#ef5350" } }}
                          style={{ borderRadius: 12, marginTop: 8 }}
                        />
                        <Text style={modal.xAxisLabel}>Day of the Week</Text>
                      </View>
                    </View>
                  )}
                  <View style={modal.dayGrid}>
                    {DAYS.map((day) => { const t = w.dailyAvgs[day]; if (t === undefined) return null; return (<View key={day} style={modal.dayCell}><Text style={modal.dayCellLabel}>{day}</Text><Text style={[modal.dayCellVal, { color: getPestColor(t) }]}>{t.toFixed(1)}</Text></View>); })}
                  </View>
                  <View style={modal.insightBox}><Text style={modal.insightText}>{getPestRecommendation(weekAvg)}</Text></View>
                </View>
              );
            })
          )}
        </ScrollView>
      </View>
    </Modal>
  );
};

const PestDetection = () => {
  const router = useRouter();
  const [currentLevel, setCurrentLevel] = useState<number>(0);
  const [weeklyAvgs, setWeeklyAvgs] = useState<{ [day: string]: number }>({});
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [sensorConnected, setSensorConnected] = useState(false);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const screenWidth = Dimensions.get("window").width;

  // ── Pest species identification state ──────────────────────────────────────
  const [pestMatches, setPestMatches] = useState<PestMatch[]>([]);
  const [identifyingPest, setIdentifyingPest] = useState(false);
  const [identificationError, setIdentificationError] = useState<string | null>(null);
  const [samplingProgress, setSamplingProgress] = useState<string | null>(null);
  const lastMetaRef = useRef<string>("");
  // ──────────────────────────────────────────────────────────────────────────

  useEffect(() => {
    Animated.loop(
      Animated.sequence([
        Animated.timing(pulseAnim, { toValue: 1.4, duration: 800, useNativeDriver: true }),
        Animated.timing(pulseAnim, { toValue: 1, duration: 800, useNativeDriver: true }),
      ])
    ).start();
  }, []);

  // ✅ Sensor connection status
  useEffect(() => {
    const deviceRef = ref(database, "soil_monitoring_system/sensors/device001");
    const unsubscribe = onValue(deviceRef, (snapshot) => {
      const data: any = snapshot.val();
      if (data) {
        const isConnected = data.status === "Connected";
        const lastSeenMs = data.lastSeen ?? 0;
        const secondsAgo = (Date.now() - lastSeenMs) / 1000;
        const isFresh = secondsAgo < 30;
        setSensorConnected(isConnected && isFresh);
      } else {
        setSensorConnected(false);
      }
    });
    return () => unsubscribe();
  }, []);

  // ✅ Pest level + daily averages
  useEffect(() => {
    const pestRef = ref(database, "soil_monitoring_system/sensors/device001/pestDetection");
    onValue(pestRef, (snapshot) => {
      const data: number = snapshot.val();
      if (data === null || data === undefined) return;
      setCurrentLevel(data);
      const dayKey = getDayKey();
      const readingsRef = ref(database, `soil_monitoring_system/sensors/device001/pestRawReadings/${dayKey}`);
      onValue(readingsRef, (snap) => {
        const existing: number[] = snap.val() || [];
        const updated = [...existing, data];
        set(readingsRef, updated);
        const dailyAvg = avg(updated);
        const dayLabel = dayKey.split("-")[2];
        const avgRef = ref(database, `soil_monitoring_system/sensors/device001/pestDailyAverages/${dayKey}`);
        set(avgRef, parseFloat(dailyAvg.toFixed(2)));
        setWeeklyAvgs((prev) => ({ ...prev, [dayLabel]: parseFloat(dailyAvg.toFixed(2)) }));
      }, { onlyOnce: true });
    });
    return () => off(pestRef);
  }, []);

  // ✅ Weekly averages
  useEffect(() => {
    const weekKey = getWeekKey();
    const avgRef = ref(database, `soil_monitoring_system/sensors/device001/pestDailyAverages`);
    onValue(avgRef, (snapshot) => {
      const all: { [key: string]: number } = snapshot.val() || {};
      const thisWeek: { [day: string]: number } = {};
      Object.entries(all).forEach(([key, value]) => { if (key.startsWith(weekKey)) { const day = key.split("-")[2]; thisWeek[day] = value; } });
      setWeeklyAvgs(thisWeek);
    });
    return () => off(avgRef);
  }, []);

  // ✅ Watch vibrationMeta — when ready=true, read all chunks and call Render API
  useEffect(() => {
    const metaRef = ref(database, "soil_monitoring_system/sensors/device001/vibrationMeta");
    const unsubscribe = onValue(metaRef, async (snapshot) => {
      const meta = snapshot.val();
      if (!meta || !meta.ready) return;

      // Avoid processing the same batch twice
      const metaKey = `${meta.totalSamples}-${meta.sampleRate}`;
      if (metaKey === lastMetaRef.current) return;
      lastMetaRef.current = metaKey;

      const { sampleRate, totalChunks } = meta;

      try {
        setIdentifyingPest(true);
        setIdentificationError(null);
        setSamplingProgress("📥 Reading vibration data from sensor...");

        // Read all chunks from Firebase and assemble full signal
        const fullSignal: number[] = [];

        for (let c = 0; c < totalChunks; c++) {
          setSamplingProgress(`📥 Loading chunk ${c + 1} of ${totalChunks}...`);
          const chunkRef = ref(database, `soil_monitoring_system/sensors/device001/vibrationSamples/chunk_${c}`);

          await new Promise<void>((resolve) => {
            onValue(chunkRef, (snap) => {
              const chunkData = snap.val();
              if (chunkData) {
                const samples = Object.keys(chunkData)
                  .sort((a, b) => parseInt(a) - parseInt(b))
                  .map((k) => chunkData[k]);
                fullSignal.push(...samples);
              }
              resolve();
            }, { onlyOnce: true });
          });
        }

        setSamplingProgress("🔬 Sending to pest identification API...");

        // Normalise from 12-bit ADC (0–4095) to m/s²
        const normalised = fullSignal.map((v) => (v - 2048) * 0.000001);

        const response = await fetch(RENDER_API_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            signal: normalised,
            sampling_rate: sampleRate,
            sensor_depth_cm: 10.0,
          }),
        });

        const result = await response.json();

        if (result.error) {
          setIdentificationError(result.error);
        } else {
          setPestMatches(result.matches || []);
        }

      } catch (err) {
        setIdentificationError("Could not reach pest identification API.");
      } finally {
        setIdentifyingPest(false);
        setSamplingProgress(null);
      }
    });
    return () => unsubscribe();
  }, []);

  const chartLabels = DAYS.filter((d) => weeklyAvgs[d] !== undefined);
  const chartValues = chartLabels.map((d) => weeklyAvgs[d]);
  const hasChartData = chartLabels.length > 0;

  const getConfidenceColor = (confidence: string): string => {
    if (confidence === "High") return "#4caf50";
    if (confidence === "Medium") return "#ffb74d";
    return "#ef5350";
  };

  return (
    <View style={styles.root}>
      <ScrollView contentContainerStyle={styles.scroll} showsVerticalScrollIndicator={false}>
        <View style={styles.topBar}>
          <TouchableOpacity style={styles.backButton} onPress={() => router.replace("/home")} activeOpacity={0.75}><Text style={styles.backButtonText}>← Home</Text></TouchableOpacity>
          <TouchableOpacity style={styles.analysisButton} onPress={() => setShowAnalysis(true)} activeOpacity={0.75}><Text style={styles.analysisButtonText}>📊 Monthly</Text></TouchableOpacity>
        </View>

        <Text style={styles.title}>🐛 Pest Detection Monitor</Text>

        <View style={[styles.sensorBadge, sensorConnected ? styles.sensorOn : styles.sensorOff]}>
          <View style={[styles.sensorDot, { backgroundColor: sensorConnected ? "#43a047" : "#e53935" }]} />
          <Text style={[styles.sensorText, { color: sensorConnected ? "#43a047" : "#e53935" }]}>
            {sensorConnected ? "Sensor Connected" : "Sensor Not Connected"}
          </Text>
        </View>

        <View style={styles.liveCard}>
          <View style={styles.liveRow}>
            <Animated.View style={[styles.liveDot, { backgroundColor: sensorConnected ? getPestColor(currentLevel) : "#3a4a5a", transform: [{ scale: pulseAnim }] }]} />
            <Text style={[styles.liveLabel, { color: sensorConnected ? getPestColor(currentLevel) : "#3a4a5a" }]}>{sensorConnected ? "LIVE" : "OFFLINE"}</Text>
          </View>
          <Text style={[styles.bigVal, { color: sensorConnected ? getPestColor(currentLevel) : "#3a4a5a" }]}>
            {sensorConnected ? getPestStatus(currentLevel) : "No Signal"}
          </Text>
          <Text style={styles.liveSubtitle}>
            {sensorConnected
              ? `Current level: ${currentLevel.toFixed(2)} — ${currentLevel === 0 ? "Clear" : currentLevel < 0.5 ? "Low risk" : "High risk"}`
              : "No sensor data available"}
          </Text>
        </View>

        {/* ── Pest Species Identification Card ─────────────────────────────── */}
        <View style={styles.speciesCard}>
          <Text style={styles.sectionTitle}>🔬 Pest Species Identification</Text>
          <Text style={styles.sectionSub}>Analysed from 5 seconds of vibration data at 10,000 Hz</Text>

          {(identifyingPest || samplingProgress) && (
            <View style={styles.loadingBox}>
              <Text style={styles.loadingText}>{samplingProgress ?? "⏳ Analysing vibration signal..."}</Text>
            </View>
          )}

          {identificationError && !identifyingPest && (
            <View style={styles.errorBox}>
              <Text style={styles.errorText}>⚠️ {identificationError}</Text>
            </View>
          )}

          {!identifyingPest && !identificationError && pestMatches.length === 0 && (
            <View style={styles.noDataBox}>
              <Text style={styles.noDataText}>Waiting for vibration sample from sensor. Results will appear automatically after each 5-second recording cycle.</Text>
            </View>
          )}

          {!identifyingPest && pestMatches.map((match, index) => (
            <View key={index} style={[styles.matchCard, { borderLeftColor: getConfidenceColor(match.confidence) }]}>
              <View style={styles.matchHeader}>
                <Text style={styles.matchRank}>#{index + 1}</Text>
                <Text style={styles.matchName}>{match.pest}</Text>
                <View style={[styles.confidenceBadge, { backgroundColor: getConfidenceColor(match.confidence) + "22", borderColor: getConfidenceColor(match.confidence) }]}>
                  <Text style={[styles.confidenceText, { color: getConfidenceColor(match.confidence) }]}>{match.confidence}</Text>
                </View>
              </View>
              <View style={styles.scoreRow}>
                <Text style={styles.scoreLabel}>Match Score</Text>
                <View style={styles.scoreBarBg}>
                  <View style={[styles.scoreBarFill, { width: `${match.score}%`, backgroundColor: getConfidenceColor(match.confidence) }]} />
                </View>
                <Text style={[styles.scoreNum, { color: getConfidenceColor(match.confidence) }]}>{match.score}/100</Text>
              </View>
              <Text style={styles.matchDetail}>🌱 <Text style={styles.matchDetailLabel}>Damage: </Text>{match.damage}</Text>
              <Text style={styles.matchDetail}>💊 <Text style={styles.matchDetailLabel}>Treatment: </Text>{match.treatment}</Text>
            </View>
          ))}
        </View>
        {/* ─────────────────────────────────────────────────────────────────── */}

        <View style={styles.chartCard}>
          <Text style={styles.sectionTitle}>📈 Weekly Daily Averages</Text>
          <Text style={styles.sectionSub}>Graph updates once per day when the daily average is calculated</Text>
          {hasChartData ? (
            <View style={styles.chartWrapper}>
              <View style={styles.yAxisLabelContainer}><Text style={styles.yAxisLabel}>Pest Level (0–1)</Text></View>
              <View style={styles.chartInner}>
                <LineChart
                  data={{ labels: chartLabels, datasets: [{ data: chartValues, color: (o = 1) => `rgba(211, 47, 47, ${o})`, strokeWidth: 3 }] }}
                  width={screenWidth - 84} height={220}
                  chartConfig={{ backgroundColor: "#0d1b2a", backgroundGradientFrom: "#0d1b2a", backgroundGradientTo: "#162232", decimalPlaces: 1, color: (o = 1) => `rgba(255,255,255,${o})`, labelColor: (o = 1) => `rgba(180,200,220,${o})`, style: { borderRadius: 16 }, propsForDots: { r: "6", strokeWidth: "2", stroke: "#ef5350" }, propsForBackgroundLines: { stroke: "#1e3248", strokeDasharray: "4,4" } }}
                  style={{ borderRadius: 14, marginTop: 8 }} bezier
                />
                <Text style={styles.xAxisLabel}>Day of the Week</Text>
              </View>
            </View>
          ) : (
            <View style={styles.noDataBox}><Text style={styles.noDataText}>Collecting readings... the chart will appear once the first daily average is ready.</Text></View>
          )}
          <View style={styles.pillRow}>
            {DAYS.map((day) => { const t = weeklyAvgs[day]; return (<View key={day} style={[styles.pill, t !== undefined && { borderColor: getPestColor(t), borderWidth: 1.5 }]}><Text style={styles.pillDay}>{day}</Text>{t !== undefined ? <Text style={[styles.pillVal, { color: getPestColor(t) }]}>{t.toFixed(1)}</Text> : <Text style={styles.pillEmpty}>—</Text>}</View>); })}
          </View>
        </View>

        <View style={styles.recCard}>
          <Text style={styles.recTitle}>💡 Recommendation</Text>
          <Text style={styles.recText}>{sensorConnected ? getPestRecommendation(currentLevel) : "Connect the sensor to receive pest detection recommendations."}</Text>
        </View>

        <View style={{ height: 32 }} />
      </ScrollView>
      <MonthlyAnalysis visible={showAnalysis} onClose={() => setShowAnalysis(false)} monthKey={getMonthKey()} />
    </View>
  );
};

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#0a1520" },
  scroll: { padding: 16, paddingTop: 52 },
  topBar: { flexDirection: "row", justifyContent: "space-between", marginBottom: 8 },
  backButton: { backgroundColor: "#1b2d42", paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, borderWidth: 1, borderColor: "#2a4060" },
  backButtonText: { color: "#90b8d8", fontWeight: "600", fontSize: 14 },
  analysisButton: { backgroundColor: "#1b3a2a", paddingHorizontal: 16, paddingVertical: 8, borderRadius: 20, borderWidth: 1, borderColor: "#2a6040" },
  analysisButtonText: { color: "#6ecfa0", fontWeight: "600", fontSize: 14 },
  title: { fontSize: 24, fontWeight: "800", textAlign: "center", color: "#d4e8f0", marginBottom: 12, letterSpacing: 0.5 },
  sensorBadge: { flexDirection: "row", alignItems: "center", alignSelf: "center", paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20, gap: 6, borderWidth: 1, marginBottom: 16 },
  sensorOn: { backgroundColor: "#0d2a1a", borderColor: "#a5d6a7" },
  sensorOff: { backgroundColor: "#2a1010", borderColor: "#ef9a9a" },
  sensorDot: { width: 8, height: 8, borderRadius: 4 },
  sensorText: { fontSize: 13, fontWeight: "700" },
  liveCard: { backgroundColor: "#111d2b", borderRadius: 20, padding: 24, alignItems: "center", marginBottom: 16, borderWidth: 1, borderColor: "#1e3248", shadowColor: "#000", shadowOpacity: 0.4, shadowRadius: 12, elevation: 6 },
  liveRow: { flexDirection: "row", alignItems: "center", marginBottom: 8 },
  liveDot: { width: 10, height: 10, borderRadius: 5, marginRight: 6 },
  liveLabel: { fontWeight: "700", fontSize: 12, letterSpacing: 1.5 },
  bigVal: { fontSize: 28, fontWeight: "900", marginTop: 4, textAlign: "center" },
  liveSubtitle: { color: "#5a7a90", fontSize: 13, marginTop: 6, textAlign: "center" },
  speciesCard: { backgroundColor: "#111d2b", borderRadius: 20, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: "#1e3248", shadowColor: "#000", shadowOpacity: 0.3, shadowRadius: 10, elevation: 5 },
  loadingBox: { backgroundColor: "#0d1520", borderRadius: 12, padding: 16, alignItems: "center", marginTop: 8 },
  loadingText: { color: "#90b8d8", fontSize: 14, textAlign: "center" },
  errorBox: { backgroundColor: "#2a1010", borderRadius: 12, padding: 16, marginTop: 8 },
  errorText: { color: "#ef9a9a", fontSize: 13 },
  matchCard: { backgroundColor: "#0d1520", borderRadius: 14, padding: 14, marginTop: 10, borderLeftWidth: 4 },
  matchHeader: { flexDirection: "row", alignItems: "center", marginBottom: 8, gap: 8 },
  matchRank: { color: "#5a7a90", fontSize: 13, fontWeight: "700", width: 24 },
  matchName: { color: "#d4e8f0", fontSize: 15, fontWeight: "700", flex: 1 },
  confidenceBadge: { paddingHorizontal: 8, paddingVertical: 3, borderRadius: 10, borderWidth: 1 },
  confidenceText: { fontSize: 11, fontWeight: "700" },
  scoreRow: { flexDirection: "row", alignItems: "center", marginBottom: 8, gap: 8 },
  scoreLabel: { color: "#5a7a90", fontSize: 11, width: 72 },
  scoreBarBg: { flex: 1, height: 6, backgroundColor: "#1e3248", borderRadius: 3, overflow: "hidden" },
  scoreBarFill: { height: 6, borderRadius: 3 },
  scoreNum: { fontSize: 12, fontWeight: "700", width: 48, textAlign: "right" },
  matchDetail: { color: "#7a9ab0", fontSize: 12, lineHeight: 18, marginTop: 2 },
  matchDetailLabel: { color: "#a0bcd0", fontWeight: "600" },
  chartCard: { backgroundColor: "#111d2b", borderRadius: 20, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: "#1e3248", shadowColor: "#000", shadowOpacity: 0.3, shadowRadius: 10, elevation: 5 },
  sectionTitle: { color: "#c8dce8", fontSize: 17, fontWeight: "700", marginBottom: 2 },
  sectionSub: { color: "#4a6070", fontSize: 12, marginBottom: 4 },
  chartWrapper: { flexDirection: "row", alignItems: "center", marginTop: 4 },
  yAxisLabelContainer: { width: 20, height: 220, justifyContent: "center", alignItems: "center" },
  yAxisLabel: { color: "#5a7a90", fontSize: 10, fontWeight: "600", transform: [{ rotate: "-90deg" }], width: 130, textAlign: "center", letterSpacing: 0.4 },
  chartInner: { flex: 1 },
  xAxisLabel: { color: "#5a7a90", fontSize: 11, fontWeight: "600", textAlign: "center", marginTop: 6, letterSpacing: 0.5 },
  noDataBox: { backgroundColor: "#0d1520", borderRadius: 12, padding: 24, alignItems: "center", marginTop: 12 },
  noDataText: { color: "#4a6070", textAlign: "center", lineHeight: 20 },
  pillRow: { flexDirection: "row", justifyContent: "space-between", marginTop: 14, flexWrap: "wrap" },
  pill: { alignItems: "center", backgroundColor: "#0d1520", borderRadius: 10, padding: 8, minWidth: 40, borderWidth: 1, borderColor: "#1e3248", marginBottom: 4 },
  pillDay: { color: "#5a7a90", fontSize: 11, fontWeight: "600" },
  pillVal: { fontSize: 14, fontWeight: "700", marginTop: 2 },
  pillEmpty: { color: "#2a4050", fontSize: 14, marginTop: 2 },
  recCard: { backgroundColor: "#111d2b", borderRadius: 20, padding: 20, borderWidth: 1, borderColor: "#1e3248", shadowColor: "#000", shadowOpacity: 0.3, shadowRadius: 10, elevation: 4 },
  recTitle: { color: "#ffffff", fontSize: 17, fontWeight: "700", marginBottom: 10 },
  recText: { color: "#a0bcd0", fontSize: 15, lineHeight: 24 },
});

const modal = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#0a1520" },
  header: { paddingTop: 52, paddingHorizontal: 20, paddingBottom: 16, backgroundColor: "#0d1b2a", borderBottomWidth: 1, borderBottomColor: "#1e3248" },
  backBtn: { alignSelf: "flex-start", paddingHorizontal: 14, paddingVertical: 7, borderRadius: 16, backgroundColor: "#1b2d42", borderWidth: 1, borderColor: "#2a4060", marginBottom: 10 },
  backBtnText: { color: "#90b8d8", fontWeight: "600", fontSize: 14 },
  title: { color: "#d4e8f0", fontSize: 22, fontWeight: "800" },
  subtitle: { color: "#4a6070", fontSize: 14, marginTop: 2 },
  scroll: { padding: 20, paddingBottom: 60 },
  cardRow: { flexDirection: "row", justifyContent: "space-between", marginBottom: 20 },
  card: { flex: 1, backgroundColor: "#111d2b", borderRadius: 14, padding: 12, alignItems: "center", marginHorizontal: 4, borderWidth: 1.5 },
  cardLabel: { color: "#5a7a90", fontSize: 11, fontWeight: "600", marginBottom: 4 },
  cardValue: { fontSize: 20, fontWeight: "800" },
  cardSub: { color: "#5a7a90", fontSize: 11, marginTop: 2 },
  weekBlock: { backgroundColor: "#111d2b", borderRadius: 16, padding: 16, marginBottom: 16, borderWidth: 1, borderColor: "#1e3248" },
  weekHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  weekTitle: { color: "#c8dce8", fontSize: 16, fontWeight: "700" },
  weekAvg: { fontSize: 16, fontWeight: "700" },
  chartWrapper: { flexDirection: "row", alignItems: "center", marginTop: 4 },
  yAxisLabelContainer: { width: 20, height: 180, justifyContent: "center", alignItems: "center" },
  yAxisLabel: { color: "#5a7a90", fontSize: 10, fontWeight: "600", transform: [{ rotate: "-90deg" }], width: 120, textAlign: "center", letterSpacing: 0.4 },
  xAxisLabel: { color: "#5a7a90", fontSize: 11, fontWeight: "600", textAlign: "center", marginTop: 6, letterSpacing: 0.5 },
  dayGrid: { flexDirection: "row", flexWrap: "wrap", marginTop: 12, gap: 6 },
  dayCell: { backgroundColor: "#0d1520", borderRadius: 8, paddingVertical: 8, paddingHorizontal: 10, alignItems: "center", minWidth: 44, borderWidth: 1, borderColor: "#1e3248" },
  dayCellLabel: { color: "#4a6070", fontSize: 11, fontWeight: "600" },
  dayCellVal: { fontSize: 15, fontWeight: "700", marginTop: 2 },
  insightBox: { backgroundColor: "#0d1520", borderRadius: 10, padding: 12, marginTop: 12, borderLeftWidth: 3, borderLeftColor: "#ef5350" },
  insightText: { color: "#90a8b8", fontSize: 13, lineHeight: 20 },
  empty: { color: "#4a6070", textAlign: "center", marginTop: 40, fontSize: 15 },
});

export default PestDetection;
