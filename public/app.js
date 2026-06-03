const ETA_API_BASE = "http://localhost:8000";

const state = {
  map: null,
  routeLine: null,
  originMarker: null,
  destinationMarker: null,
  modelInfo: null,
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
  resolveButton: document.getElementById("resolve-button"),
  sampleButton: document.getElementById("sample-button"),
  clearButton: document.getElementById("clear-button"),
  distanceValue: document.getElementById("distance-value"),
  durationValue: document.getElementById("duration-value"),
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

async function fetchEtaModelInfo() {
  const response = await fetch(`${ETA_API_BASE}/api/eta/model-info`);
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
  elements.vehicleValue.textContent = state.modelInfo?.model_name || "-";
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
  const metrics = modelInfo.holdout_metrics || {};
  const quantileMetrics = Array.isArray(modelInfo.quantile_holdout_metrics)
    ? modelInfo.quantile_holdout_metrics
    : [];

  const items = [
    `Point model: ${modelInfo.model_name || "-"}`,
    `Selection: ${modelInfo.selection_policy || "-"}`,
    `CV-best: ${modelInfo.cv_best_model_name || modelInfo.model_name || "-"}`,
    `MAE holdout: ${Number(metrics.test_mae_seconds || 0).toFixed(2)} sec`,
    ...quantileMetrics.map((metric) => {
      const label = `P${Math.round(Number(metric.quantile) * 100)}`;
      return `${label} coverage: ${Number(metric.test_coverage || 0).toFixed(3)} / target ${Number(
        metric.target_coverage || 0,
      ).toFixed(2)}`;
    }),
  ];

  elements.stepsList.innerHTML = "";
  elements.stepsCount.textContent = `${items.length} metrics`;
  for (const text of items) {
    const item = document.createElement("li");
    item.innerHTML = `<strong>${text}</strong>`;
    elements.stepsList.appendChild(item);
  }
}

function renderPrediction(data) {
  const prediction = data.prediction;
  const point = prediction.point;
  const quantiles = prediction.quantiles || {};

  elements.distanceValue.textContent = formatDistance(data.route?.distance_meters);
  elements.durationValue.textContent = formatEta(point.minutes);
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

    const data = await postEtaPrediction({
      departure_time: elements.departureTime.value,
    });

    renderPrediction(data);
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
  renderDiagnostics(modelInfo);
  setDefaultDepartureTime();
  resetSummary();
  setStatus("Ready");
}

elements.form.addEventListener("submit", calculateRoute);
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
