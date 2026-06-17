// Nếu đang chạy ở local thì gọi localhost, nếu trên Vercel thì gọi backend Render
const ETA_API_BASE = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" 
  ? "http://localhost:8000" 
  : "https://https://eta-fastapi-backend.onrender.com"; // TODO: Đổi thành URL backend thực tế của bạn trên Render

// ── Model metadata (from experiment results) ──────────────────────────────
const MODEL_META = {
  ratio_time_bin: {
    label: "Ratio Time-bin",
    description:
      "Phương pháp tốt nhất về P95. Nhân baseline ETA với hệ số median ratio theo từng khung giờ.",
    testMae: 38.02,
    testP95: 97.28,
    maeImprovementPct: -0.16,
    p95ImprovementPct: 21.39,
    requiresBaseline: true,
    targetType: "ratio",
    formula: "pred_ETA = baseline_ETA × median(actual/baseline | time_bin)",
  },
  log_ratio_global: {
    label: "Log-Ratio Global",
    description:
      "Phương pháp tốt nhất về MAE. Học log(actual/baseline) toàn cục rồi exp() để ra hệ số nhân.",
    testMae: 37.24,
    testP95: 108.04,
    maeImprovementPct: 1.89,
    p95ImprovementPct: 12.7,
    requiresBaseline: true,
    targetType: "log_ratio",
    formula: "pred_ETA = baseline_ETA × exp(median(log(actual/baseline)))",
  },
  additive_global: {
    label: "Additive Global",
    description:
      "Phương pháp đơn giản, ổn định nhất. Cộng thêm hằng số residual toàn cục vào baseline ETA.",
    testMae: 37.25,
    testP95: 107.95,
    maeImprovementPct: 1.86,
    p95ImprovementPct: 12.76,
    requiresBaseline: true,
    targetType: "additive",
    formula: "pred_ETA = baseline_ETA + median(actual − baseline)",
  },
};

// Approximate time-bin ratios from training data
const TIME_BIN_RATIOS = {
  early_morning: 1.058,
  morning_peak: 1.145,
  off_peak_day: 1.092,
  evening_peak: 1.13,
  late_evening: 1.072,
  night: 1.04,
};
const GLOBAL_RATIO = 1.1009;
const GLOBAL_ADDITIVE = 15.795;

function getTimeBin(h) {
  if (h >= 4 && h < 7) return "early_morning";
  if (h >= 7 && h < 10) return "morning_peak";
  if (h >= 10 && h < 15) return "off_peak_day";
  if (h >= 15 && h < 19) return "evening_peak";
  if (h >= 19 && h < 22) return "late_evening";
  return "night";
}

// ── State ──────────────────────────────────────────────────────────────────
const state = {
  map: null,
  routePolyline: null,        // full route geo polyline
  segmentPolyline: null,      // highlighted segment between origin/dest stations
  stationMarkers: [],         // all station markers on route
  originMarker: null,
  destinationMarker: null,
  busRouteData: null,         // raw route.json dt
  currentDirection: "Go",     // Go or Re
  routeGeo: [],               // Geo polyline for current direction
  stations: [],               // stations for current direction
  originStationIdx: null,
  destinationStationIdx: null,
  selectedModelId: "ratio_time_bin",
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const el = {
  form: document.getElementById("route-form"),
  busRoute: document.getElementById("bus-route"),
  busDirection: document.getElementById("bus-direction"),
  originStation: document.getElementById("origin-station"),
  destinationStation: document.getElementById("destination-station"),
  originDisplay: document.getElementById("origin-display"),
  destinationDisplay: document.getElementById("destination-display"),
  originLat: document.getElementById("origin-lat"),
  originLng: document.getElementById("origin-lng"),
  destinationLat: document.getElementById("destination-lat"),
  destinationLng: document.getElementById("destination-lng"),
  originResolveState: document.getElementById("origin-resolve-state"),
  destinationResolveState: document.getElementById("destination-resolve-state"),
  etaModel: document.getElementById("eta-model"),
  departureTime: document.getElementById("departure-time"),
  sampleButton: document.getElementById("sample-button"),
  distanceValue: document.getElementById("distance-value"),
  durationValue: document.getElementById("duration-value"),
  baselineValue: document.getElementById("baseline-value"),
  methodValue: document.getElementById("method-value"),
  etaHourValue: document.getElementById("eta-hour-value"),
  errorMessage: document.getElementById("error-message"),
  stepsList: document.getElementById("steps-list"),
  stepsCount: document.getElementById("steps-count"),
  statusPill: document.getElementById("status-pill"),
  mapTitle: document.getElementById("map-title"),
  modelDescription: document.getElementById("model-description"),
  infoMae: document.getElementById("info-mae"),
  infoP95: document.getElementById("info-p95"),
  infoImprovement: document.getElementById("info-improvement"),
};

// ── Helpers ────────────────────────────────────────────────────────────────
function setStatus(t) { el.statusPill.textContent = t; }
function setError(msg) {
  if (!msg) { el.errorMessage.hidden = true; el.errorMessage.textContent = ""; return; }
  el.errorMessage.hidden = false; el.errorMessage.textContent = msg;
}
function fmtCoord(v) { return Number.isFinite(Number(v)) ? Number(v).toFixed(6) : "-"; }
function fmtDist(m) {
  const v = Number(m); if (!Number.isFinite(v)) return "-";
  return v < 1000 ? `${Math.round(v)} m` : `${(v / 1000).toFixed(2)} km`;
}
function fmtEta(sec) {
  const v = Number(sec); if (!Number.isFinite(v)) return "-";
  return `${(v / 60).toFixed(1)} phút`;
}
function setDefaultDepartureTime() {
  const now = new Date(); now.setSeconds(0, 0);
  const p = (v) => String(v).padStart(2, "0");
  el.departureTime.value =
    [now.getFullYear(), p(now.getMonth() + 1), p(now.getDate())].join("-") +
    `T${p(now.getHours())}:${p(now.getMinutes())}`;
}

// ── Map helpers ────────────────────────────────────────────────────────────
function mkIcon(color, size = 14) {
  return L.divIcon({
    className: "custom-marker",
    html: `<span style="display:block;width:${size}px;height:${size}px;border-radius:999px;border:3px solid white;background:${color};box-shadow:0 6px 12px rgba(0,0,0,0.25)"></span>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

function clearAll() {
  if (state.routePolyline) { state.routePolyline.remove(); state.routePolyline = null; }
  if (state.segmentPolyline) { state.segmentPolyline.remove(); state.segmentPolyline = null; }
  for (const m of state.stationMarkers) m.remove();
  state.stationMarkers = [];
  if (state.originMarker) { state.originMarker.remove(); state.originMarker = null; }
  if (state.destinationMarker) { state.destinationMarker.remove(); state.destinationMarker = null; }
}

function drawFullRoute(geo) {
  if (state.routePolyline) state.routePolyline.remove();
  const latlngs = geo.map((p) => [p.Lat, p.Lng]);
  state.routePolyline = L.polyline(latlngs, {
    color: "#ef5b2a", weight: 4, opacity: 0.45, lineCap: "round", dashArray: "6,8",
  }).addTo(state.map);
  state.map.fitBounds(state.routePolyline.getBounds(), { padding: [60, 60] });
}

// ── Geo polyline helpers ───────────────────────────────────────────────────
function haversineDist(lat1, lon1, lat2, lon2) {
  const R = 6371e3;
  const p1 = (lat1 * Math.PI) / 180, p2 = (lat2 * Math.PI) / 180;
  const dp = ((lat2 - lat1) * Math.PI) / 180, dl = ((lon2 - lon1) * Math.PI) / 180;
  const a = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

function polylineLength(geoArray) {
  let d = 0;
  for (let i = 0; i < geoArray.length - 1; i++) {
    d += haversineDist(geoArray[i][0], geoArray[i][1], geoArray[i + 1][0], geoArray[i + 1][1]);
  }
  return d;
}
// Find the index in the Geo polyline closest to a given lat/lng.
function nearestGeoIndex(geo, lat, lng) {
  let best = 0;
  let bestDist = Infinity;
  for (let i = 0; i < geo.length; i++) {
    const dLat = geo[i].Lat - lat;
    const dLng = geo[i].Lng - lng;
    const d = dLat * dLat + dLng * dLng;
    if (d < bestDist) { bestDist = d; best = i; }
  }
  return best;
}

// Extract the Geo polyline segment between two stations.
function extractGeoSegment(geo, stationA, stationB) {
  const idxA = nearestGeoIndex(geo, stationA.Geo.Lat, stationA.Geo.Lng);
  const idxB = nearestGeoIndex(geo, stationB.Geo.Lat, stationB.Geo.Lng);
  const lo = Math.min(idxA, idxB);
  const hi = Math.max(idxA, idxB);
  return geo.slice(lo, hi + 1).map((p) => [p.Lat, p.Lng]);
}

function drawStationMarkers(stations) {
  for (const m of state.stationMarkers) m.remove();
  state.stationMarkers = [];
  for (const s of stations) {
    const m = L.marker([s.Geo.Lat, s.Geo.Lng], { icon: mkIcon("#6875f5", 10) })
      .addTo(state.map)
      .bindPopup(`<strong>${s.Name}</strong><br/>Mã: ${s.Code}`);
    state.stationMarkers.push(m);
  }
}

function highlightOriginDest() {
  if (state.originMarker) state.originMarker.remove();
  if (state.destinationMarker) state.destinationMarker.remove();
  if (state.segmentPolyline) state.segmentPolyline.remove();

  const oIdx = state.originStationIdx;
  const dIdx = state.destinationStationIdx;

  if (oIdx !== null) {
    const s = state.stations[oIdx];
    state.originMarker = L.marker([s.Geo.Lat, s.Geo.Lng], { icon: mkIcon("#0f9d58", 20), zIndexOffset: 1000 })
      .addTo(state.map).bindPopup(`<strong>Xuất phát:</strong> ${s.Name}`);
    el.originDisplay.textContent = s.Name;
    el.originLat.textContent = fmtCoord(s.Geo.Lat);
    el.originLng.textContent = fmtCoord(s.Geo.Lng);
    el.originResolveState.textContent = "Đã chọn";
  }
  if (dIdx !== null) {
    const s = state.stations[dIdx];
    state.destinationMarker = L.marker([s.Geo.Lat, s.Geo.Lng], { icon: mkIcon("#c63b35", 20), zIndexOffset: 1000 })
      .addTo(state.map).bindPopup(`<strong>Điểm đến:</strong> ${s.Name}`);
    el.destinationDisplay.textContent = s.Name;
    el.destinationLat.textContent = fmtCoord(s.Geo.Lat);
    el.destinationLng.textContent = fmtCoord(s.Geo.Lng);
    el.destinationResolveState.textContent = "Đã chọn";
  }

  // Draw thick segment between origin and destination using actual Geo polyline
  if (oIdx !== null && dIdx !== null && state.routeGeo.length > 0) {
    const sA = state.stations[oIdx];
    const sB = state.stations[dIdx];
    const segLatLngs = extractGeoSegment(state.routeGeo, sA, sB);
    state.segmentPolyline = L.polyline(segLatLngs, {
      color: "#0f9d58", weight: 6, opacity: 0.9, lineCap: "round",
    }).addTo(state.map);
    state.map.fitBounds(state.segmentPolyline.getBounds(), { padding: [80, 80] });
  }
}

// ── Populate station dropdowns ─────────────────────────────────────────────
function populateStations(stations) {
  state.stations = stations;
  state.originStationIdx = null;
  state.destinationStationIdx = null;

  for (const sel of [el.originStation, el.destinationStation]) {
    sel.innerHTML = '<option value="" disabled selected>-- Chọn trạm --</option>';
    stations.forEach((s, i) => {
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = `${i + 1}. ${s.Name}`;
      sel.appendChild(opt);
    });
  }
  resetStationCards();
}

function resetStationCards() {
  el.originDisplay.textContent = "-";
  el.originLat.textContent = "-";
  el.originLng.textContent = "-";
  el.originResolveState.textContent = "Chờ";
  el.destinationDisplay.textContent = "-";
  el.destinationLat.textContent = "-";
  el.destinationLng.textContent = "-";
  el.destinationResolveState.textContent = "Chờ";
}

// ── Model info panel ───────────────────────────────────────────────────────
function renderModelInfo(modelId) {
  const m = MODEL_META[modelId]; if (!m) return;
  el.modelDescription.textContent = m.description;
  el.infoMae.textContent = `${m.testMae.toFixed(2)} s`;
  el.infoP95.textContent = `${m.testP95.toFixed(2)} s`;
  const sign = m.maeImprovementPct >= 0 ? "+" : "";
  el.infoImprovement.textContent = `MAE ${sign}${m.maeImprovementPct.toFixed(2)}%`;

  const items = [
    `Phương pháp: ${m.label}`,
    `Công thức: ${m.formula}`,
    `Target type: ${m.targetType}`,
    `Test MAE: ${m.testMae.toFixed(2)} s`,
    `Test P95: ${m.testP95.toFixed(2)} s`,
    `MAE vs Vietmap: ${sign}${m.maeImprovementPct.toFixed(2)}%`,
    `P95 vs Vietmap: +${m.p95ImprovementPct.toFixed(2)}%`,
  ];
  el.stepsList.innerHTML = "";
  el.stepsCount.textContent = `${items.length} items`;
  for (const txt of items) {
    const li = document.createElement("li");
    li.innerHTML = `<strong>${txt}</strong>`;
    el.stepsList.appendChild(li);
  }
}

function resetSummary() {
  el.distanceValue.textContent = "-";
  el.durationValue.textContent = "-";
  el.baselineValue.textContent = "-";
  el.methodValue.textContent = MODEL_META[state.selectedModelId]?.label || "-";
  el.etaHourValue.textContent = "-";
}

// ── API calls ──────────────────────────────────────────────────────────────
async function fetchConfig() {
  const r = await fetch("/api/config"); if (!r.ok) throw new Error("Config failed"); return r.json();
}

async function fetchBusRoutes() {
  const r = await fetch("/api/bus-routes"); if (!r.ok) throw new Error("Bus routes failed"); return r.json();
}

async function fetchVietmapBaseline(origin, destination, departureTime) {
  const r = await fetch("/api/route", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      origin: { lat: origin.Lat, lng: origin.Lng },
      destination: { lat: destination.Lat, lng: destination.Lng },
      vehicle: "car",
      departureTime,
      alternative: false,
    }),
  });
  const data = await r.json();
  if (!r.ok) throw new Error(data.details || data.error || "Vietmap baseline failed.");
  const secs = Number(data.summary?.durationMs) / 1000;
  if (!Number.isFinite(secs) || secs <= 0) throw new Error("Invalid Vietmap duration.");
  return { ...data, baselineEtaSecs: secs };
}

async function postEtaPrediction(payload) {
  const r = await fetch(`${ETA_API_BASE}/api/eta/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await r.json();
  if (!r.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((i) => i.msg).join(", ") : data.detail;
    throw new Error(detail || data.error || "ETA prediction failed.");
  }
  return data;
}

// ── Local correction fallback ──────────────────────────────────────────────
function applyLocalCorrection(baselineSecs, departureTime, modelId) {
  const hour = new Date(departureTime).getHours();
  let corrected, desc;
  if (modelId === "ratio_time_bin") {
    const bin = getTimeBin(hour);
    const ratio = TIME_BIN_RATIOS[bin];
    corrected = baselineSecs * ratio;
    desc = `Ratio Time-bin [${bin}: ×${ratio}]`;
  } else if (modelId === "log_ratio_global") {
    corrected = baselineSecs * GLOBAL_RATIO;
    desc = `Log-Ratio Global [×${GLOBAL_RATIO}]`;
  } else {
    corrected = baselineSecs + GLOBAL_ADDITIVE;
    desc = `Additive Global [+${GLOBAL_ADDITIVE}s]`;
  }
  return { corrected, desc, hour };
}

// ── Form submit ────────────────────────────────────────────────────────────
async function handlePredict(event) {
  event.preventDefault();
  setError(""); setStatus("Đang tính");

  try {
    if (state.originStationIdx === null || state.destinationStationIdx === null) {
      throw new Error("Vui lòng chọn trạm xuất phát và trạm đến.");
    }
    if (state.originStationIdx === state.destinationStationIdx) {
      throw new Error("Trạm xuất phát và trạm đến phải khác nhau.");
    }
    if (!el.departureTime.value) throw new Error("Vui lòng chọn thời gian khởi hành.");

    const modelId = el.etaModel.value;
    state.selectedModelId = modelId;
    const meta = MODEL_META[modelId];

    const oStation = state.stations[state.originStationIdx];
    const dStation = state.stations[state.destinationStationIdx];

    // 1. Fetch Vietmap baseline with only origin + destination (2 points, no zigzag)
    setStatus("Vietmap");
    const baseline = await fetchVietmapBaseline(oStation.Geo, dStation.Geo, el.departureTime.value);

    // Draw the actual bus route Geo segment on the map (not Vietmap's routed path)
    if (state.routeGeo.length > 0) {
      if (state.segmentPolyline) state.segmentPolyline.remove();
      const segLatLngs = extractGeoSegment(state.routeGeo, oStation, dStation);
      state.segmentPolyline = L.polyline(segLatLngs, {
        color: "#0f9d58", weight: 6, opacity: 0.9, lineCap: "round",
      }).addTo(state.map);
      state.map.fitBounds(state.segmentPolyline.getBounds(), { padding: [80, 80] });
    }

    // 2. Predict ETA piecewise (C->D + D->E + ...)
    setStatus("Predicting (từng trạm)");
    let etaSecs = 0;
    let modelLabel = meta?.label || modelId;
    let hourLabel = "-";
    
    const lo = Math.min(state.originStationIdx, state.destinationStationIdx);
    const hi = Math.max(state.originStationIdx, state.destinationStationIdx);
    
    const fullGeoSegment = extractGeoSegment(state.routeGeo, oStation, dStation);
    const totalGeoDist = polylineLength(fullGeoSegment) || 1; // avoid division by zero

    const promises = [];
    for (let i = lo; i < hi; i++) {
      const s1 = state.stations[i];
      const s2 = state.stations[i + 1];
      const segGeo = extractGeoSegment(state.routeGeo, s1, s2);
      const segDist = polylineLength(segGeo);
      
      const weight = segDist / totalGeoDist;
      const segBaselineSecs = baseline.baselineEtaSecs * weight;

      const payload = {
        departure_time: el.departureTime.value,
        model_id: modelId,
        baseline_eta_secs: segBaselineSecs,
      };

      promises.push(
        postEtaPrediction(payload)
          .then((data) => {
            if (i === lo) hourLabel = `Hour ${data.prediction?.hour ?? new Date(el.departureTime.value).getHours()}`;
            return data.prediction?.point?.seconds ?? data.prediction?.point?.minutes * 60;
          })
          .catch(() => {
            const local = applyLocalCorrection(segBaselineSecs, el.departureTime.value, modelId);
            if (i === lo) { hourLabel = `Hour ${local.hour} (local)`; modelLabel = local.desc; }
            return local.corrected;
          })
      );
    }
    
    const segEtas = await Promise.all(promises);
    etaSecs = segEtas.reduce((acc, val) => acc + val, 0);

    // 3. Render
    el.distanceValue.textContent = fmtDist(baseline.summary?.distanceMeters);
    el.durationValue.textContent = fmtEta(etaSecs);
    el.baselineValue.textContent = fmtEta(baseline.baselineEtaSecs);
    el.methodValue.textContent = modelLabel;
    el.etaHourValue.textContent = hourLabel;
    setStatus("Ready");
  } catch (err) {
    setStatus("Error");
    resetSummary();
    setError(err instanceof Error ? err.message : "Prediction failed.");
  }
}

// ── Direction change ───────────────────────────────────────────────────────
function onDirectionChange() {
  const dir = el.busDirection.value;
  state.currentDirection = dir;
  const dirData = state.busRouteData?.[dir];
  if (!dirData) return;

  clearAll();
  state.routeGeo = dirData.Geo;  // store for segment extraction
  drawFullRoute(dirData.Geo);
  populateStations(dirData.Station);
  drawStationMarkers(dirData.Station);
  resetSummary();
  setError("");
}

// ── Initialization ─────────────────────────────────────────────────────────
async function initialize() {
  const [config, busData] = await Promise.all([fetchConfig(), fetchBusRoutes()]);

  state.map = L.map("map", { zoomControl: false }).setView(
    [config.mapCenter.lat, config.mapCenter.lng], config.mapZoom,
  );
  L.control.zoom({ position: "bottomright" }).addTo(state.map);
  L.tileLayer(config.tileLayer.url, { attribution: config.tileLayer.attribution, maxZoom: 19 }).addTo(state.map);

  // Load route data into memory
  const dt = busData.dt;
  state.busRouteData = dt;

  // Populate route dropdown (single route for PoC)
  el.busRoute.innerHTML = "";
  const opt = document.createElement("option");
  opt.value = dt.Code;
  opt.textContent = `${dt.Code} — ${dt.Name} (${dt.Enterprise})`;
  opt.selected = true;
  el.busRoute.appendChild(opt);

  el.mapTitle.textContent = `${dt.Code}: ${dt.Name}`;

  // Initialize with "Go" direction
  onDirectionChange();

  setDefaultDepartureTime();
  renderModelInfo(state.selectedModelId);
  resetSummary();
  setStatus("Ready");
}

// ── Event listeners ────────────────────────────────────────────────────────
el.form.addEventListener("submit", handlePredict);

el.busDirection.addEventListener("change", onDirectionChange);

el.originStation.addEventListener("change", () => {
  state.originStationIdx = el.originStation.value !== "" ? Number(el.originStation.value) : null;
  highlightOriginDest();
});

el.destinationStation.addEventListener("change", () => {
  state.destinationStationIdx = el.destinationStation.value !== "" ? Number(el.destinationStation.value) : null;
  highlightOriginDest();
});

el.etaModel.addEventListener("change", () => {
  state.selectedModelId = el.etaModel.value;
  renderModelInfo(state.selectedModelId);
  resetSummary();
  setError("");
});

el.sampleButton.addEventListener("click", () => {
  setDefaultDepartureTime();
  resetSummary();
  setError("");
  setStatus("Ready");
});

initialize().catch((err) => {
  setStatus("Error");
  setError(err instanceof Error ? err.message : "Failed to initialize.");
});
