const state = {
  map: null,
  routeLine: null,
  originMarker: null,
  destinationMarker: null,
  locations: {
    origin: {
      requestId: 0,
      address: "",
      point: null,
      display: "",
      refId: "",
      dirty: true,
    },
    destination: {
      requestId: 0,
      address: "",
      point: null,
      display: "",
      refId: "",
      dirty: true,
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
  stepsList: document.getElementById("steps-list"),
  stepsCount: document.getElementById("steps-count"),
  statusPill: document.getElementById("status-pill"),
  errorMessage: document.getElementById("error-message"),
};

function locationElements(kind) {
  if (kind === "origin") {
    return {
      addressInput: elements.originAddress,
      latValue: elements.originLat,
      lngValue: elements.originLng,
      displayValue: elements.originDisplay,
      resolveState: elements.originResolveState,
    };
  }

  return {
    addressInput: elements.destinationAddress,
    latValue: elements.destinationLat,
    lngValue: elements.destinationLng,
    displayValue: elements.destinationDisplay,
    resolveState: elements.destinationResolveState,
  };
}

async function fetchConfig() {
  const response = await fetch("/api/config");
  if (!response.ok) {
    throw new Error("Failed to load map config.");
  }
  return response.json();
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

function normalizeRouteGeometryData(data) {
  const directGeometry = Array.isArray(data?.geometry) ? data.geometry : [];
  if (directGeometry.length) {
    return directGeometry;
  }

  const rawCoordinates = data?.raw?.paths?.[0]?.points?.coordinates;
  if (!Array.isArray(rawCoordinates) || !rawCoordinates.length) {
    return [];
  }

  return rawCoordinates
    .map((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) {
        return null;
      }

      const first = Number(pair[0]);
      const second = Number(pair[1]);
      if (!Number.isFinite(first) || !Number.isFinite(second)) {
        return null;
      }

      if (Math.abs(first) > 40 && Math.abs(second) <= 40) {
        return [second, first];
      }

      return [first, second];
    })
    .filter(Boolean);
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

function clearMarker(kind) {
  const marker = kind === "origin" ? state.originMarker : state.destinationMarker;
  if (!marker) {
    return;
  }

  marker.remove();
  if (kind === "origin") {
    state.originMarker = null;
  } else {
    state.destinationMarker = null;
  }
}

function syncLocationCard(kind) {
  const entry = state.locations[kind];
  const scopedElements = locationElements(kind);

  if (!entry.point) {
    scopedElements.displayValue.textContent = entry.address || "-";
    scopedElements.latValue.textContent = "-";
    scopedElements.lngValue.textContent = "-";
    return;
  }

  scopedElements.displayValue.textContent = entry.display || entry.address;
  scopedElements.latValue.textContent = entry.point.lat.toFixed(6);
  scopedElements.lngValue.textContent = entry.point.lng.toFixed(6);
}

function setResolveState(kind, text) {
  locationElements(kind).resolveState.textContent = text;
}

function setLocationDraft(kind, address) {
  const entry = state.locations[kind];
  const nextAddress = address.trim();

  entry.address = nextAddress;
  entry.point = null;
  entry.display = "";
  entry.refId = "";
  entry.dirty = true;

  syncLocationCard(kind);
  setResolveState(kind, nextAddress ? "Needs sync" : "Waiting");
  clearMarker(kind);
  clearRouteVisuals();
  resetSummary();

  if (elements.statusPill.textContent === "Ready") {
    setStatus("Idle");
  }
}

function applyResolvedLocation(kind, resolved) {
  const entry = state.locations[kind];
  const point = resolved.point;

  entry.address = resolved.query || entry.address;
  entry.point = point;
  entry.display = resolved.display || entry.address;
  entry.refId = resolved.refId || "";
  entry.dirty = false;

  syncLocationCard(kind);
  setResolveState(kind, "Ready");
  drawMarker(kind, point, entry.display);
}

function resetSummary() {
  elements.distanceValue.textContent = "-";
  elements.durationValue.textContent = "-";
  elements.vehicleValue.textContent = "-";
  elements.stepsCount.textContent = "0 steps";
  elements.stepsList.innerHTML = "";
}

function clearRouteVisuals() {
  if (state.routeLine) {
    state.routeLine.remove();
    state.routeLine = null;
  }
}

function clearAll() {
  elements.form.reset();
  resetSummary();
  setError("");
  setStatus("Idle");
  clearRouteVisuals();
  clearMarker("origin");
  clearMarker("destination");

  state.locations.origin = {
    requestId: 0,
    address: "",
    point: null,
    display: "",
    refId: "",
    dirty: true,
  };
  state.locations.destination = {
    requestId: 0,
    address: "",
    point: null,
    display: "",
    refId: "",
    dirty: true,
  };

  syncLocationCard("origin");
  syncLocationCard("destination");
  setResolveState("origin", "Waiting");
  setResolveState("destination", "Waiting");
}

async function resolveLocation(kind, options = {}) {
  const entry = state.locations[kind];
  const scopedElements = locationElements(kind);
  const address = scopedElements.addressInput.value.trim();

  if (!address) {
    throw new Error(`Please enter a ${kind} address.`);
  }

  if (!entry.dirty && entry.point && entry.address === address) {
    return entry;
  }

  entry.requestId += 1;
  const requestId = entry.requestId;
  entry.address = address;
  setResolveState(kind, options.silent ? "Syncing" : "Resolving");

  const payload = {
    address,
  };

  if (kind === "destination" && state.locations.origin.point) {
    payload.focus = state.locations.origin.point;
  }

  const resolved = await postJson("/api/resolve-location", payload);
  if (requestId !== entry.requestId) {
    return entry;
  }

  applyResolvedLocation(kind, resolved);
  return entry;
}

async function resolveBothLocations() {
  await resolveLocation("origin");
  await resolveLocation("destination");

  const originPoint = state.locations.origin.point;
  const destinationPoint = state.locations.destination.point;

  if (originPoint && destinationPoint) {
    const bounds = L.latLngBounds(
      [originPoint.lat, originPoint.lng],
      [destinationPoint.lat, destinationPoint.lng],
    );
    state.map.fitBounds(bounds, { padding: [48, 48] });
  }
}

function populateSample() {
  elements.originAddress.value = "197 Tran Phu, Phuong 4, Quan 5, Thanh pho Ho Chi Minh";
  elements.destinationAddress.value = "292 Dinh Bo Linh, Phuong 26, Quan Binh Thanh, Thanh pho Ho Chi Minh";
  elements.vehicle.value = "car";
  elements.capacityKg.value = "";
  elements.departureTime.value = "";
  elements.alternative.checked = false;

  setLocationDraft("origin", elements.originAddress.value);
  setLocationDraft("destination", elements.destinationAddress.value);
}

function renderSteps(instructions) {
  elements.stepsList.innerHTML = "";
  elements.stepsCount.textContent = `${instructions.length} steps`;

  for (const step of instructions.slice(0, 12)) {
    const item = document.createElement("li");
    item.innerHTML = `
      <strong>${step.text || "Continue"}</strong>
      <div>${step.distanceLabel} · ${step.durationLabel}</div>
    `;
    elements.stepsList.appendChild(item);
  }
}

function renderRoute(geometry, summary, instructions, resolvedLocations) {
  clearRouteVisuals();

  state.routeLine = L.polyline(geometry, {
    color: "#ef5b2a",
    weight: 6,
    opacity: 0.92,
    lineCap: "round",
    lineJoin: "round",
  }).addTo(state.map);

  state.map.fitBounds(state.routeLine.getBounds(), {
    padding: [40, 40],
  });

  elements.distanceValue.textContent = summary.distanceLabel;
  elements.durationValue.textContent = summary.durationLabel;
  elements.vehicleValue.textContent = summary.vehicle;
  renderSteps(instructions);

  if (resolvedLocations?.origin?.point) {
    applyResolvedLocation("origin", resolvedLocations.origin);
  }
  if (resolvedLocations?.destination?.point) {
    applyResolvedLocation("destination", resolvedLocations.destination);
  }
}

async function calculateRoute(event) {
  event.preventDefault();
  setError("");
  setStatus("Resolving");

  try {
    await resolveBothLocations();
    setStatus("Routing");

    const payload = {
      origin: {
        lat: state.locations.origin.point.lat,
        lng: state.locations.origin.point.lng,
        address: state.locations.origin.address,
      },
      destination: {
        lat: state.locations.destination.point.lat,
        lng: state.locations.destination.point.lng,
        address: state.locations.destination.address,
      },
      vehicle: elements.vehicle.value,
      alternative: elements.alternative.checked,
    };

    if (elements.capacityKg.value) {
      payload.capacityKg = Number(elements.capacityKg.value);
    }

    if (elements.departureTime.value.trim()) {
      payload.departureTime = elements.departureTime.value.trim();
    }

    const data = await postJson("/api/route", payload);
    const geometry = normalizeRouteGeometryData(data);
    if (!geometry.length) {
      throw new Error("Vietmap returned no route geometry.");
    }

    setStatus("Ready");
    renderRoute(geometry, data.summary, data.instructions || [], data.resolvedLocations);
  } catch (error) {
    setStatus("Error");
    resetSummary();
    clearRouteVisuals();
    setError(error instanceof Error ? error.message : "Route lookup failed.");
  }
}

async function handleResolveClick() {
  setError("");
  setStatus("Resolving");

  try {
    await resolveBothLocations();
    setStatus("Ready");
  } catch (error) {
    setStatus("Error");
    setError(error instanceof Error ? error.message : "Address resolution failed.");
  }
}

function registerAddressInput(kind) {
  const scopedElements = locationElements(kind);

  scopedElements.addressInput.addEventListener("input", (event) => {
    const value = event.currentTarget.value;
    setLocationDraft(kind, value);
  });

  scopedElements.addressInput.addEventListener("blur", async () => {
    if (!scopedElements.addressInput.value.trim()) {
      return;
    }

    try {
      await resolveLocation(kind, { silent: true });
    } catch {
      setResolveState(kind, "Retry");
    }
  });
}

async function initialize() {
  const config = await fetchConfig();

  state.map = L.map("map", {
    zoomControl: false,
  }).setView([config.mapCenter.lat, config.mapCenter.lng], config.mapZoom);

  L.control.zoom({ position: "bottomright" }).addTo(state.map);

  L.tileLayer(config.tileLayer.url, {
    attribution: config.tileLayer.attribution,
    maxZoom: 19,
  }).addTo(state.map);

  elements.vehicle.value = config.vehicle;
  syncLocationCard("origin");
  syncLocationCard("destination");
  populateSample();
}

elements.form.addEventListener("submit", calculateRoute);
elements.resolveButton.addEventListener("click", handleResolveClick);
elements.sampleButton.addEventListener("click", populateSample);
elements.clearButton.addEventListener("click", clearAll);

registerAddressInput("origin");
registerAddressInput("destination");

initialize().catch((error) => {
  setStatus("Error");
  setError(error instanceof Error ? error.message : "Failed to initialize map.");
});
