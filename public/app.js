const ETA_API_BASE = "http://localhost:8000";

const state = {
  map: null,
  routeLine: null,
  originMarker: null,
  destinationMarker: null,
  modelInfo: null,
  availableModels: [],
  selectedModelId: null,
  fixedRoute: null,
  locations: {
    origin: {
      requestId: 0,
      address: "",
      point: null,
      display: "",
      refId: "",
      dirty: false,
    },
    destination: {
      requestId: 0,
      address: "",
      point: null,
      display: "",
      refId: "",
      dirty: false,
    },
  },
};

const elements = {
  form: document.getElementById("route-form"),
  originAddress: document.getElementById("origin-address"),
  destinationAddress: document.getElementById("destination-address"),
  originLat: document.getElementById("origin-lat"),
  originLng: document.getElementById("origin-lng"),
  destinationLat: document.getElementById("destination-lat"),
  destinationLng: document.getElementById("destination-lng"),
  originDisplay: document.getElementById("origin-display"),
  destinationDisplay: document.getElementById("destination-display"),
  originResolveState: document.getElementById("origin-resolve-state"),
  destinationResolveState: document.getElementById("destination-resolve-state"),
  vehicle: document.getElementById("vehicle"),
  capacityKg: document.getElementById("capacity-kg"),
  departureTime: document.getElementById("departure-time"),
  alternative: document.getElementById("alternative"),
  etaModel: document.getElementById("eta-model"),
  resolveButton: document.getElementById("resolve-button"),
  sampleButton: document.getElementById("sample-button"),
  clearButton: document.getElementById("clear-button"),
  distanceValue: document.getElementById("distance-value"),
  durationValue: document.getElementById("duration-value"),
  baselineValue: document.getElementById("baseline-value"),
  vehicleValue: document.getElementById("vehicle-value"),
  etaHourValue: document.getElementById("eta-hour-value"),
  etaP50Value: document.getElementById("eta-p50-value"),
  etaP85Value: document.getElementById("eta-p85-value"),
  etaP90Value: document.getElementById("eta-p90-value"),
  stepsList: document.getElementById("steps-list"),
  stepsCount: document.getElementById("steps-count"),
  statusPill: document.getElementById("status-pill"),
  errorMessage: document.getElementById("error-message"),
};

async function fetchConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error("Failed to load map config.");
  }
  return response.json();
}

async function fetchEtaModelInfo(modelId) {
  const suffix = modelId ? `?model_id=${encodeURIComponent(modelId)}` : "";
  const response = await fetch(`${ETA_API_BASE}/api/eta/model-info${suffix}`);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || data.error || "Failed to load ETA model.");
  }
  return data;
}

async function postEtaPrediction(payload) {
  const response = await fetch(`${ETA_API_BASE}/api/eta/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(", ") : data.detail;
    throw new Error(detail || data.error || "ETA prediction failed.");
  }
  return data;
}

async function fetchVietmapBaseline(departureTime) {
  const origin = state.locations.origin.point;
  const destination = state.locations.destination.point;
  if (!origin || !destination) {
    throw new Error("Fixed route coordinates are not ready.");
  }

  const response = await fetch("/api/route", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      origin,
      destination,
      vehicle: elements.vehicle?.value || "car",
      departureTime,
      alternative: false,
    }),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.details || data.error || "Failed to fetch Vietmap baseline ETA.");
  }
  const baselineEtaSecs = Number(data.summary?.durationMs) / 1000;
  if (!Number.isFinite(baselineEtaSecs) || baselineEtaSecs <= 0) {
    throw new Error("Vietmap route response did not include a valid duration.");
  }
  return {
    ...data,
    baselineEtaSecs,
  };
}

function setStatus(text) {
  elements.statusPill.textContent = text;
}

function setError(message) {
  if (!message) {
    elements.errorMessage.hidden = true;
    elements.errorMessage.textContent = "";
    return;
  }
  elements.errorMessage.hidden = false;
  elements.errorMessage.textContent = message;
}

function formatCoordinate(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(6) : "-";
}

function formatDistance(meters) {
  const value = Number(meters);
  if (!Number.isFinite(value)) {
    return "-";
  }
  if (value < 1000) {
    return `${Math.round(value)} m`;
  }
  return `${(value / 1000).toFixed(2)} km`;
}

function formatEta(minutes) {
  const value = Number(minutes);
  if (!Number.isFinite(value)) {
    return "-";
  }
  return `${value.toFixed(2)} min`;
}

function selectedModel() {
  return state.availableModels.find((model) => model.id === state.selectedModelId) || null;
}

function populateModelSelect(models, selectedModelId) {
  elements.etaModel.innerHTML = "";
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = model.available ? model.label : `${model.label} (unavailable)`;
    option.disabled = !model.available;
    elements.etaModel.appendChild(option);
  }
  elements.etaModel.value = selectedModelId;
}

function markerIcon(color) {
  return L.divIcon({
    className: "custom-marker",
    html: `<span style="display:block;width:18px;height:18px;border-radius:999px;border:3px solid white;background:${color};box-shadow:0 10px 16px rgba(0,0,0,0.18)"></span>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  });
}

function drawMarker(kind, point, display) {
  const marker = kind === "origin" ? state.originMarker : state.destinationMarker;
  const icon = markerIcon(kind === "origin" ? "#0f9d58" : "#c63b35");
  const popup = display || `${point.lat.toFixed(6)}, ${point.lng.toFixed(6)}`;

  if (marker) {
    marker.setLatLng([point.lat, point.lng]).setIcon(icon).bindPopup(popup);
    return;
  }

  const createdMarker = L.marker([point.lat, point.lng], { icon }).addTo(state.map).bindPopup(popup);
  if (kind === "origin") {
    state.originMarker = createdMarker;
  } else {
    state.destinationMarker = createdMarker;
  }
}

function clearRouteVisuals() {
  if (state.routeLine) {
    state.routeLine.remove();
    state.routeLine = null;
  }
}

function syncLocationCard(kind) {
  const entry = state.locations[kind];
  const prefix = kind === "origin" ? "origin" : "destination";
  elements[`${prefix}Display`].textContent = entry.display || "-";
  elements[`${prefix}Lat`].textContent = entry.point ? formatCoordinate(entry.point.lat) : "-";
  elements[`${prefix}Lng`].textContent = entry.point ? formatCoordinate(entry.point.lng) : "-";
}

function setFixedLocation(kind, point, display) {
  state.locations[kind] = {
    requestId: 0,
    address: display,
    point,
    display,
    refId: "",
    dirty: false,
  };

  if (kind === "origin") {
    elements.originAddress.value = display;
    elements.originResolveState.textContent = "Fixed";
  } else {
    elements.destinationAddress.value = display;
    elements.destinationResolveState.textContent = "Fixed";
  }

  syncLocationCard(kind);
  drawMarker(kind, point, display);
}

function renderFixedRoute(route) {
  clearRouteVisuals();
  const origin = route.origin;
  const destination = route.destination;
  if (!origin?.lat || !origin?.lng || !destination?.lat || !destination?.lng) {
    return;
  }

  setFixedLocation("origin", { lat: Number(origin.lat), lng: Number(origin.lng) }, origin.label || "Origin station");
  setFixedLocation(
    "destination",
    { lat: Number(destination.lat), lng: Number(destination.lng) },
    destination.label || "Destination station",
  );

  state.routeLine = L.polyline(
    [
      [Number(origin.lat), Number(origin.lng)],
      [Number(destination.lat), Number(destination.lng)],
    ],
    {
      color: "#ef5b2a",
      weight: 6,
      opacity: 0.92,
      lineCap: "round",
      lineJoin: "round",
    },
  ).addTo(state.map);

  state.map.fitBounds(state.routeLine.getBounds(), {
    padding: [48, 48],
  });
}

function resetSummary() {
  elements.distanceValue.textContent = state.fixedRoute ? formatDistance(state.fixedRoute.distance_meters) : "-";
  elements.durationValue.textContent = "-";
  elements.baselineValue.textContent = "-";
  elements.vehicleValue.textContent = selectedModel()?.label || state.modelInfo?.model_name || "-";
  elements.etaHourValue.textContent = "-";
  elements.etaP50Value.textContent = "-";
  elements.etaP85Value.textContent = "-";
  elements.etaP90Value.textContent = "-";
}

function setDefaultDepartureTime() {
  const now = new Date();
  now.setSeconds(0, 0);
  const pad = (value) => String(value).padStart(2, "0");
  elements.departureTime.value = [
    now.getFullYear(),
    pad(now.getMonth() + 1),
    pad(now.getDate()),
  ].join("-") + `T${pad(now.getHours())}:${pad(now.getMinutes())}`;
}

function renderDiagnostics(modelInfo) {
  const selected = modelInfo.selected_model || selectedModel() || {};
  const metrics = selected.metrics || modelInfo.holdout_metrics || {};
  const quantileMetrics = Array.isArray(modelInfo.quantile_holdout_metrics)
    ? modelInfo.quantile_holdout_metrics
    : [];

  const testMae = metrics.test_mae ?? metrics.test_mae_seconds;
  const testP95 = metrics.test_p95 ?? metrics.test_p95_abs_error;
  const improvement = metrics.mae_improvement_pct ?? metrics.test_mae_improvement_pct;
  const items = [
    `Selected: ${selected.label || modelInfo.model_name || "-"}`,
    `Type: ${selected.model_type || "legacy"}`,
    `Target: ${selected.target_type || modelInfo.target_column || "-"}`,
    `Needs Vietmap baseline: ${selected.requires_baseline ? "yes" : "no"}`,
  ];

  if (Number.isFinite(Number(testMae))) {
    items.push(`Test MAE: ${Number(testMae).toFixed(2)} sec`);
  }
  if (Number.isFinite(Number(testP95))) {
    items.push(`Test P95: ${Number(testP95).toFixed(2)} sec`);
  }
  if (Number.isFinite(Number(improvement))) {
    items.push(`MAE improvement vs Vietmap: ${Number(improvement).toFixed(2)}%`);
  }
  if (!items.some((item) => item.startsWith("Test MAE")) && Number.isFinite(Number(metrics.test_mae_seconds))) {
    items.push(`MAE holdout: ${Number(metrics.test_mae_seconds).toFixed(2)} sec`);
  }
  items.push(
    ...quantileMetrics.map((metric) => {
      const label = `P${Math.round(Number(metric.quantile) * 100)}`;
      return `${label} coverage: ${Number(metric.test_coverage || 0).toFixed(3)} / target ${Number(
        metric.target_coverage || 0,
      ).toFixed(2)}`;
    }),
  );

  elements.stepsList.innerHTML = "";
  elements.stepsCount.textContent = `${items.length} metrics`;
  for (const text of items) {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${text}</strong>`;
    elements.stepsList.appendChild(item);
  }
}

function renderPrediction(data, baselineRoute) {
  const prediction = data.prediction;
  const point = prediction.point;
  const quantiles = prediction.quantiles || {};
  const baselineMinutes = prediction.baseline?.minutes ?? baselineRoute?.baselineEtaSecs / 60;
  const routeDistance = baselineRoute?.summary?.distanceMeters ?? data.route?.distance_meters;

  elements.distanceValue.textContent = formatDistance(routeDistance);
  elements.durationValue.textContent = formatEta(point.minutes);
  elements.baselineValue.textContent = formatEta(baselineMinutes);
  elements.vehicleValue.textContent = data.model_name || "-";
  elements.etaHourValue.textContent = `Hour ${prediction.hour}`;
  elements.etaP50Value.textContent = formatEta(quantiles.p50?.minutes);
  elements.etaP85Value.textContent = formatEta(quantiles.p85?.minutes);
  elements.etaP90Value.textContent = formatEta(quantiles.p90?.minutes);
}

async function calculateRoute(event) {
  event.preventDefault();
  setError("");
  setStatus("Predicting");

  try {
    if (!elements.departureTime.value) {
      throw new Error("Please select a departure time.");
    }

    const model = selectedModel();
    if (!model) {
      throw new Error("Please select a prediction model.");
    }

    let baselineRoute = null;
    const payload = {
      departure_time: elements.departureTime.value,
      model_id: model.id,
    };

    if (model.requires_baseline) {
      setStatus("Vietmap");
      baselineRoute = await fetchVietmapBaseline(elements.departureTime.value);
      payload.baseline_eta_secs = baselineRoute.baselineEtaSecs;
    }

    setStatus("Predicting");
    const data = await postEtaPrediction(payload);

    renderPrediction(data, baselineRoute);
    setStatus("Ready");
  } catch (error) {
    setStatus("Error");
    resetSummary();
    setError(error instanceof Error ? error.message : "ETA prediction failed.");
  }
}

async function initialize() {
  const [config, modelInfo] = await Promise.all([fetchConfig(), fetchEtaModelInfo()]);
  state.modelInfo = modelInfo;
  state.availableModels = modelInfo.available_models || [];
  state.selectedModelId = modelInfo.selected_model_id || state.availableModels.find((model) => model.available)?.id || null;
  state.fixedRoute = modelInfo.route;

  state.map = L.map("map", {
    zoomControl: false,
  }).setView([config.mapCenter.lat, config.mapCenter.lng], config.mapZoom);

  L.control.zoom({ position: "bottomright" }).addTo(state.map);

  L.tileLayer(config.tileLayer.url, {
    attribution: config.tileLayer.attribution,
    maxZoom: 19,
  }).addTo(state.map);

  renderFixedRoute(modelInfo.route);
  populateModelSelect(state.availableModels, state.selectedModelId);
  renderDiagnostics(modelInfo);
  setDefaultDepartureTime();
  resetSummary();
  setStatus("Ready");
}

elements.form.addEventListener("submit", calculateRoute);
elements.etaModel.addEventListener("change", async () => {
  state.selectedModelId = elements.etaModel.value;
  setError("");
  setStatus("Loading");
  try {
    const modelInfo = await fetchEtaModelInfo(state.selectedModelId);
    state.modelInfo = modelInfo;
    state.availableModels = modelInfo.available_models || state.availableModels;
    renderDiagnostics(modelInfo);
    resetSummary();
    setStatus("Ready");
  } catch (error) {
    setStatus("Error");
    setError(error instanceof Error ? error.message : "Failed to load model metadata.");
  }
});
elements.sampleButton.addEventListener("click", () => {
  setDefaultDepartureTime();
  resetSummary();
  setError("");
  setStatus("Ready");
});

initialize().catch((error) => {
  setStatus("Error");
  setError(error instanceof Error ? error.message : "Failed to initialize map.");
});
