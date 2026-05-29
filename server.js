"use strict";

const http = require("node:http");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const { URL } = require("node:url");
let h3 = null;
try {
  h3 = require("h3-js");
} catch {
  h3 = null;
}

const rootDir = __dirname;
const publicDir = path.join(rootDir, "public");

function loadEnv(filePath) {
  const result = {};
  if (!fs.existsSync(filePath)) {
    return result;
  }

  const raw = fs.readFileSync(filePath, "utf8");
  for (const line of raw.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith("#")) {
      continue;
    }

    const separatorIndex = trimmed.indexOf("=");
    if (separatorIndex === -1) {
      continue;
    }

    const key = trimmed.slice(0, separatorIndex).trim();
    const value = trimmed.slice(separatorIndex + 1).trim().replace(/^"(.*)"$/, "$1");
    result[key] = value;
  }
  return result;
}

function parseHolidayDates(rawValue) {
  return new Set(
    String(rawValue || "")
      .split(",")
      .map((value) => value.trim())
      .filter(Boolean),
  );
}

const env = {
  ...loadEnv(path.join(rootDir, ".env")),
  ...process.env,
};

const config = {
  port: Number(env.PORT || 3000),
  vietmapApiKey: env.VIETMAP_API_KEY || "",
  vietmapRouteUrl: env.VIETMAP_ROUTE_URL || "https://maps.vietmap.vn/api/route/v3",
  vietmapSearchUrl: env.VIETMAP_SEARCH_URL || "https://maps.vietmap.vn/api/search/v4",
  vietmapPlaceUrl: env.VIETMAP_PLACE_URL || "https://maps.vietmap.vn/api/place/v4",
  vietmapDisplayType: Number(env.VIETMAP_DISPLAY_TYPE || 5),
  vietmapDefaultVehicle: env.VIETMAP_DEFAULT_VEHICLE || "car",
  vietmapTileApiKey: env.VIETMAP_TILE_API_KEY || "",
  vietmapTileUrlTemplate: env.VIETMAP_TILE_URL_TEMPLATE || "",
  publicMapTileUrl: env.PUBLIC_MAP_TILE_URL || "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
  publicMapTileAttribution: env.PUBLIC_MAP_TILE_ATTRIBUTION || "&copy; OpenStreetMap contributors",
  mapCenterLat: Number(env.MAP_CENTER_LAT || 10.7769),
  mapCenterLng: Number(env.MAP_CENTER_LNG || 106.7009),
  mapZoom: Number(env.MAP_ZOOM || 12),
  etaServiceUrl: env.ETA_SERVICE_URL || "http://localhost:8000",
  etaTimeoutMs: Number(env.ETA_TIMEOUT_MS || 3500),
  etaH3Resolution: Number(env.ETA_H3_RESOLUTION || 9),
  etaDefaultTrafficLevel: env.ETA_DEFAULT_TRAFFIC_LEVEL || "medium",
  etaDefaultIsRaining: String(env.ETA_DEFAULT_IS_RAINING || "").toLowerCase() === "true",
  etaDefaultRainLevel: env.ETA_DEFAULT_RAIN_LEVEL || "none",
  etaDefaultWeatherCondition: env.ETA_DEFAULT_WEATHER_CONDITION || "clear",
  etaHolidayDates: parseHolidayDates(env.ETA_HOLIDAY_DATES),
};

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

const TRAFFIC_LEVELS = ["low", "medium", "high", "severe"];
const RAIN_LEVELS = ["none", "light", "moderate", "heavy", "very_heavy"];
const WEATHER_CONDITIONS = ["clear", "cloudy", "rain", "storm", "fog"];

function sendJson(response, statusCode, payload) {
  response.writeHead(statusCode, { "Content-Type": "application/json; charset=utf-8" });
  response.end(JSON.stringify(payload));
}

function sendText(response, statusCode, message) {
  response.writeHead(statusCode, { "Content-Type": "text/plain; charset=utf-8" });
  response.end(message);
}

function parseBody(request) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    request.on("data", (chunk) => chunks.push(chunk));
    request.on("end", () => {
      try {
        const body = Buffer.concat(chunks).toString("utf8");
        resolve(body ? JSON.parse(body) : {});
      } catch (error) {
        reject(error);
      }
    });
    request.on("error", reject);
  });
}

function isFiniteCoordinate(value, min, max) {
  return Number.isFinite(value) && value >= min && value <= max;
}

function normalizePoint(input) {
  if (!input || typeof input !== "object") {
    return null;
  }

  const lat = Number(input.lat);
  const lng = Number(input.lng);
  if (!isFiniteCoordinate(lat, -90, 90) || !isFiniteCoordinate(lng, -180, 180)) {
    return null;
  }

  return {
    lat: Number(lat.toFixed(6)),
    lng: Number(lng.toFixed(6)),
  };
}

function normalizeText(value) {
  return typeof value === "string" ? value.trim() : "";
}

function normalizeDepartureTimeInput(value) {
  const raw = normalizeText(value);
  if (!raw) {
    return "";
  }

  const parsed = new Date(raw);
  if (Number.isNaN(parsed.getTime())) {
    throw new Error("Departure time must be a valid date/time.");
  }

  return parsed.toISOString();
}

function formatDuration(milliseconds) {
  const totalMinutes = Math.max(1, Math.round(milliseconds / 60000));
  const hours = Math.floor(totalMinutes / 60);
  const minutes = totalMinutes % 60;
  if (!hours) {
    return `${totalMinutes} min`;
  }
  return `${hours} hr ${minutes} min`;
}

function formatDistance(meters) {
  if (meters < 1000) {
    return `${Math.round(meters)} m`;
  }
  return `${(meters / 1000).toFixed(1)} km`;
}

function pad2(value) {
  return String(value).padStart(2, "0");
}

function formatLocalDate(date) {
  return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}`;
}

function dayNameFromDate(date) {
  return ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][date.getDay()];
}

function isRushHour(hourOfDay) {
  return [7, 8, 17, 18, 19].includes(hourOfDay);
}

function toRadians(value) {
  return (value * Math.PI) / 180;
}

function haversineMeters(lat1, lng1, lat2, lng2) {
  const radius = 6371000;
  const dLat = toRadians(lat2 - lat1);
  const dLng = toRadians(lng2 - lng1);
  const a =
    Math.sin(dLat / 2) ** 2 +
    Math.cos(toRadians(lat1)) * Math.cos(toRadians(lat2)) * Math.sin(dLng / 2) ** 2;
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return radius * c;
}

function normalizeEnum(value, allowedValues, fallbackValue) {
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (allowedValues.includes(normalized)) {
    return normalized;
  }
  return fallbackValue;
}

function parseBoolean(value, fallbackValue) {
  if (typeof value === "boolean") {
    return value;
  }
  const normalized = typeof value === "string" ? value.trim().toLowerCase() : "";
  if (["true", "1", "yes"].includes(normalized)) {
    return true;
  }
  if (["false", "0", "no"].includes(normalized)) {
    return false;
  }
  return fallbackValue;
}

function computeH3Cell(point) {
  if (h3 && typeof h3.latLngToCell === "function") {
    return h3.latLngToCell(point.lat, point.lng, config.etaH3Resolution);
  }
  return `h3_${point.lat.toFixed(4)}_${point.lng.toFixed(4)}`;
}

function deriveTrafficLevelFromAnnotations(congestion) {
  const values = Array.isArray(congestion)
    ? congestion.map((value) => Number(value)).filter((value) => Number.isFinite(value))
    : [];

  if (!values.length) {
    return normalizeEnum(config.etaDefaultTrafficLevel, TRAFFIC_LEVELS, "medium");
  }

  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  if (average >= 3.2) {
    return "severe";
  }
  if (average >= 2.2) {
    return "high";
  }
  if (average >= 1.1) {
    return "medium";
  }
  return "low";
}

function buildEtaFeatures(payload, origin, destination, routePath) {
  const departureDate = payload?.departureTime ? new Date(payload.departureTime) : new Date();
  const safeDate = Number.isNaN(departureDate.getTime()) ? new Date() : departureDate;
  const hourOfDay = safeDate.getHours();
  const dayOfWeek = dayNameFromDate(safeDate);
  const dateKey = formatLocalDate(safeDate);

  const defaultTrafficLevel = deriveTrafficLevelFromAnnotations(routePath?.annotations?.congestion);
  const trafficLevel = normalizeEnum(
    payload?.traffic_level || payload?.trafficLevel,
    TRAFFIC_LEVELS,
    defaultTrafficLevel,
  );

  const isRainingDefault =
    config.etaDefaultIsRaining || ["rain", "storm"].includes(config.etaDefaultWeatherCondition);
  const isRaining = parseBoolean(payload?.is_raining ?? payload?.isRaining, isRainingDefault);
  const rainLevel = normalizeEnum(
    payload?.rain_level || payload?.rainLevel,
    RAIN_LEVELS,
    normalizeEnum(config.etaDefaultRainLevel, RAIN_LEVELS, isRaining ? "light" : "none"),
  );
  const weatherCondition = normalizeEnum(
    payload?.weather_condition || payload?.weatherCondition,
    WEATHER_CONDITIONS,
    normalizeEnum(
      config.etaDefaultWeatherCondition,
      WEATHER_CONDITIONS,
      isRaining ? "rain" : "clear",
    ),
  );

  const distanceValue = Number(routePath?.distance);
  const timeValue = Number(routePath?.time);
  const baselineDistanceMeters = Number.isFinite(distanceValue)
    ? distanceValue
    : haversineMeters(origin.lat, origin.lng, destination.lat, destination.lng);
  const baselineEtaSecs = Number.isFinite(timeValue)
    ? Math.max(1, Math.round(timeValue / 1000))
    : Math.max(1, Math.round(baselineDistanceMeters / 6));

  return {
    origin_h3: computeH3Cell(origin),
    destination_h3: computeH3Cell(destination),
    origin_lng: origin.lng,
    origin_lat: origin.lat,
    destination_lng: destination.lng,
    destination_lat: destination.lat,
    hour_of_day: hourOfDay,
    is_rush_hour: isRushHour(hourOfDay),
    day_of_week: dayOfWeek,
    is_weekend: dayOfWeek === "Sat" || dayOfWeek === "Sun",
    is_holiday: config.etaHolidayDates.has(dateKey),
    haversine_distance_meters: haversineMeters(
      origin.lat,
      origin.lng,
      destination.lat,
      destination.lng,
    ),
    baseline_distance_meters: baselineDistanceMeters,
    traffic_level: trafficLevel,
    is_raining: isRaining,
    rain_level: rainLevel,
    weather_condition: weatherCondition,
    baseline_eta_secs: baselineEtaSecs,
  };
}

async function fetchEtaPrediction(features) {
  if (!config.etaServiceUrl) {
    return { error: "ETA service URL is not configured." };
  }

  const requestUrl = new URL("/predict", config.etaServiceUrl);
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), config.etaTimeoutMs);

  try {
    const response = await fetch(requestUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ records: [features] }),
      signal: controller.signal,
    });

    const data = await response.json();
    if (!response.ok) {
      return {
        error: data?.detail || data?.error || "ETA service returned an error.",
      };
    }

    const prediction = Array.isArray(data?.predictions) ? data.predictions[0] : null;
    if (!prediction || !Number.isFinite(Number(prediction.eta_seconds))) {
      return { error: "ETA service did not return a prediction." };
    }

    return {
      etaSeconds: Number(prediction.eta_seconds),
      modelName: data?.model_name || "xgboost",
      modelUri: data?.model_uri || "",
    };
  } catch (error) {
    const message =
      error && error.name === "AbortError"
        ? "ETA service timed out."
        : error instanceof Error
          ? error.message
          : "ETA service request failed.";
    return { error: message };
  } finally {
    clearTimeout(timeoutId);
  }
}

function normalizeGeometry(points) {
  const rawPoints = Array.isArray(points)
    ? points
    : Array.isArray(points?.coordinates)
      ? points.coordinates
      : [];

  if (!rawPoints.length) {
    return [];
  }

  return rawPoints
    .map((pair) => {
      if (!Array.isArray(pair) || pair.length < 2) {
        return null;
      }

      const first = Number(pair[0]);
      const second = Number(pair[1]);
      if (!Number.isFinite(first) || !Number.isFinite(second)) {
        return null;
      }

      // Vietmap docs contain both [lat,lng] and [lng,lat] wording, so normalize defensively.
      if (Math.abs(first) > 40 && Math.abs(second) <= 40) {
        return [second, first];
      }
      return [first, second];
    })
    .filter(Boolean);
}

async function fetchJsonFromUpstream(url) {
  let upstreamResponse;
  try {
    upstreamResponse = await fetch(url, {
      method: "GET",
      headers: {
        Accept: "application/json",
      },
    });
  } catch (error) {
    throw new Error(error instanceof Error ? error.message : String(error));
  }

  let data;
  try {
    data = await upstreamResponse.json();
  } catch {
    throw new Error("Upstream returned a non-JSON response.");
  }

  if (!upstreamResponse.ok) {
    const message = data?.message || data?.error || `Upstream request failed with ${upstreamResponse.status}.`;
    throw new Error(message);
  }

  return data;
}

async function resolveRefId(refId) {
  const upstream = new URL(config.vietmapPlaceUrl);
  upstream.searchParams.set("apikey", config.vietmapApiKey);
  upstream.searchParams.set("refid", refId);

  const detail = await fetchJsonFromUpstream(upstream);
  const point = normalizePoint({
    lat: detail?.lat,
    lng: detail?.lng,
  });

  if (!point) {
    throw new Error("Vietmap Place API did not return usable coordinates.");
  }

  return {
    point,
    display: normalizeText(detail?.display) || normalizeText(detail?.address) || refId,
    detail,
    refId,
  };
}

async function searchAddress(address, focusPoint) {
  const upstream = new URL(config.vietmapSearchUrl);
  upstream.searchParams.set("apikey", config.vietmapApiKey);
  upstream.searchParams.set("text", address);
  upstream.searchParams.set("display_type", String(config.vietmapDisplayType));

  if (focusPoint) {
    upstream.searchParams.set("focus", `${focusPoint.lat},${focusPoint.lng}`);
  }

  const matches = await fetchJsonFromUpstream(upstream);
  if (!Array.isArray(matches) || !matches.length) {
    throw new Error(`No Vietmap geocoding result found for "${address}".`);
  }

  const bestMatch = matches[0];
  const refId = normalizeText(bestMatch?.ref_id);
  if (!refId) {
    throw new Error("Vietmap geocoding result is missing ref_id.");
  }

  const resolved = await resolveRefId(refId);
  return {
    ...resolved,
    query: address,
    candidate: bestMatch,
  };
}

function extractInputAddress(input) {
  if (typeof input === "string") {
    return normalizeText(input);
  }
  if (!input || typeof input !== "object") {
    return "";
  }
  return normalizeText(input.address) || normalizeText(input.display) || normalizeText(input.text);
}

function extractFocusPoint(input) {
  if (!input || typeof input !== "object") {
    return null;
  }
  return normalizePoint(input.focus) || normalizePoint(input.bias) || normalizePoint(input);
}

async function resolveLocationInput(input, fallbackFocus) {
  const directPoint = normalizePoint(input);
  if (directPoint) {
    return {
      point: directPoint,
      display: "",
      detail: null,
      refId: "",
      source: "coordinates",
    };
  }

  const inputAddress = extractInputAddress(input);
  const inputRefId = input && typeof input === "object" ? normalizeText(input.refId) : "";
  const focusPoint = extractFocusPoint(input) || fallbackFocus || null;

  if (inputRefId) {
    const resolved = await resolveRefId(inputRefId);
    return {
      ...resolved,
      source: "refId",
      query: inputAddress,
    };
  }

  if (!inputAddress) {
    throw new Error("Location input must include either coordinates or an address.");
  }

  const resolved = await searchAddress(inputAddress, focusPoint);
  return {
    ...resolved,
    source: "address",
  };
}

function serializeResolvedLocation(resolved) {
  return {
    point: resolved.point,
    display: resolved.display,
    refId: resolved.refId,
    query: resolved.query || "",
    source: resolved.source,
    detail: resolved.detail,
  };
}

async function handleResolveLocation(request, response) {
  if (!config.vietmapApiKey) {
    sendJson(response, 500, { error: "Missing VIETMAP_API_KEY in .env" });
    return;
  }

  let payload;
  try {
    payload = await parseBody(request);
  } catch {
    sendJson(response, 400, { error: "Request body must be valid JSON." });
    return;
  }

  try {
    const resolved = await resolveLocationInput(payload, {
      lat: config.mapCenterLat,
      lng: config.mapCenterLng,
    });
    sendJson(response, 200, serializeResolvedLocation(resolved));
  } catch (error) {
    sendJson(response, 422, {
      error: error instanceof Error ? error.message : "Failed to resolve address.",
    });
  }
}

async function handleRoute(request, response) {
  if (!config.vietmapApiKey) {
    sendJson(response, 500, {
      error: "Missing VIETMAP_API_KEY in .env",
    });
    return;
  }

  let payload;
  try {
    payload = await parseBody(request);
  } catch {
    sendJson(response, 400, { error: "Request body must be valid JSON." });
    return;
  }

  const vehicle = ["car", "motorcycle", "truck"].includes(payload.vehicle)
    ? payload.vehicle
    : config.vietmapDefaultVehicle;

  let resolvedOrigin;
  let resolvedDestination;
  try {
    resolvedOrigin = await resolveLocationInput(payload.origin, {
      lat: config.mapCenterLat,
      lng: config.mapCenterLng,
    });
    resolvedDestination = await resolveLocationInput(
      payload.destination,
      resolvedOrigin.point || {
        lat: config.mapCenterLat,
        lng: config.mapCenterLng,
      },
    );
  } catch (error) {
    sendJson(response, 422, {
      error: error instanceof Error ? error.message : "Failed to resolve one of the locations.",
    });
    return;
  }

  const origin = resolvedOrigin.point;
  const destination = resolvedDestination.point;

  const upstream = new URL(config.vietmapRouteUrl);
  upstream.searchParams.set("apikey", config.vietmapApiKey);
  upstream.searchParams.append("point", `${origin.lat},${origin.lng}`);
  upstream.searchParams.append("point", `${destination.lat},${destination.lng}`);
  upstream.searchParams.set("vehicle", vehicle);
  upstream.searchParams.set("points_encoded", "false");
  upstream.searchParams.set("annotations", "congestion,congestion_distance");

  if (vehicle === "truck" && Number.isFinite(Number(payload.capacityKg))) {
    upstream.searchParams.set("capacity", String(Math.max(1, Number(payload.capacityKg))));
  }

  if (payload.departureTime) {
    let departureTime;
    try {
      departureTime = normalizeDepartureTimeInput(payload.departureTime);
    } catch (error) {
      sendJson(response, 400, {
        error: error instanceof Error ? error.message : "Departure time is invalid.",
      });
      return;
    }

    upstream.searchParams.set("time", departureTime);
  }

  if (payload.alternative === true) {
    upstream.searchParams.set("alternative", "true");
  }

  let data;
  try {
    data = await fetchJsonFromUpstream(upstream);
  } catch (error) {
    sendJson(response, 502, {
      error: "Failed to reach Vietmap Route API.",
      details: error instanceof Error ? error.message : String(error),
    });
    return;
  }

  if (data.code !== "OK" || !Array.isArray(data.paths) || !data.paths.length) {
    sendJson(response, 502, {
      error: "Vietmap Route API did not return a usable route.",
      vietmap: data,
    });
    return;
  }

  const primaryPath = data.paths[0];
  const geometry = normalizeGeometry(primaryPath.points);

  const etaFeatures = buildEtaFeatures(payload, origin, destination, primaryPath);
  const etaResult = await fetchEtaPrediction(etaFeatures);
  const etaPayload = etaResult.error
    ? null
    : {
        seconds: etaResult.etaSeconds,
        minutes: etaResult.etaSeconds / 60,
        label: formatDuration(etaResult.etaSeconds * 1000),
        modelName: etaResult.modelName,
        modelUri: etaResult.modelUri,
      };

  const responsePayload = {
    summary: {
      vehicle,
      distanceMeters: primaryPath.distance,
      durationMs: primaryPath.time,
      distanceLabel: formatDistance(primaryPath.distance),
      durationLabel: formatDuration(primaryPath.time),
      congestionSegments: Array.isArray(primaryPath.annotations?.congestion)
        ? primaryPath.annotations.congestion.length
        : 0,
    },
    geometry,
    bounds: primaryPath.bbox || null,
    instructions: Array.isArray(primaryPath.instructions)
      ? primaryPath.instructions.map((instruction, index) => ({
          id: index + 1,
          text: instruction.text,
          streetName: instruction.street_name,
          distanceMeters: instruction.distance,
          distanceLabel: formatDistance(instruction.distance),
          durationMs: instruction.time,
          durationLabel: formatDuration(instruction.time),
        }))
      : [],
    resolvedLocations: {
      origin: serializeResolvedLocation(resolvedOrigin),
      destination: serializeResolvedLocation(resolvedDestination),
    },
    raw: data,
  };

  if (etaPayload) {
    responsePayload.eta = etaPayload;
  }
  if (etaResult.error) {
    responsePayload.etaError = etaResult.error;
  }
  if (payload.debugEta) {
    responsePayload.etaFeatures = etaFeatures;
  }

  sendJson(response, 200, responsePayload);
}

async function handleTileProxy(url, response) {
  if (!config.vietmapTileApiKey || !config.vietmapTileUrlTemplate) {
    sendJson(response, 404, { error: "Vietmap tile proxy is not configured." });
    return;
  }

  const match = url.pathname.match(/^\/api\/tiles\/(\d+)\/(\d+)\/(\d+)\.png$/);
  if (!match) {
    sendJson(response, 404, { error: "Tile path is invalid." });
    return;
  }

  const [, z, x, y] = match;
  const upstreamUrl = config.vietmapTileUrlTemplate
    .replace("{z}", z)
    .replace("{x}", x)
    .replace("{y}", y);
  const resolvedUrl = new URL(upstreamUrl);
  resolvedUrl.searchParams.set("apikey", config.vietmapTileApiKey);

  let upstreamResponse;
  try {
    upstreamResponse = await fetch(resolvedUrl);
  } catch (error) {
    sendJson(response, 502, {
      error: "Failed to reach Vietmap tile server.",
      details: error instanceof Error ? error.message : String(error),
    });
    return;
  }

  if (!upstreamResponse.ok) {
    sendJson(response, 502, {
      error: "Vietmap tile server returned an error.",
      status: upstreamResponse.status,
    });
    return;
  }

  const image = Buffer.from(await upstreamResponse.arrayBuffer());
  response.writeHead(200, {
    "Content-Type": upstreamResponse.headers.get("content-type") || "image/png",
    "Cache-Control": "public, max-age=86400",
  });
  response.end(image);
}

async function serveStatic(url, response) {
  let targetPath = url.pathname === "/" ? "/index.html" : url.pathname;
  targetPath = path.normalize(targetPath).replace(/^(\.\.[/\\])+/, "");
  const filePath = path.join(publicDir, targetPath);

  if (!filePath.startsWith(publicDir)) {
    sendText(response, 403, "Forbidden");
    return;
  }

  try {
    const content = await fsp.readFile(filePath);
    const extension = path.extname(filePath).toLowerCase();
    response.writeHead(200, {
      "Content-Type": mimeTypes[extension] || "application/octet-stream",
    });
    response.end(content);
  } catch (error) {
    if (error && error.code === "ENOENT") {
      sendText(response, 404, "Not Found");
      return;
    }
    sendText(response, 500, "Failed to serve file.");
  }
}

const server = http.createServer(async (request, response) => {
  if (!request.url) {
    sendText(response, 400, "Invalid request.");
    return;
  }

  const url = new URL(request.url, `http://${request.headers.host || "localhost"}`);

  if (request.method === "GET" && url.pathname === "/api/config") {
    sendJson(response, 200, {
      mapCenter: {
        lat: config.mapCenterLat,
        lng: config.mapCenterLng,
      },
      mapZoom: config.mapZoom,
      vehicle: config.vietmapDefaultVehicle,
      tileLayer: config.vietmapTileApiKey && config.vietmapTileUrlTemplate
        ? {
            url: "/api/tiles/{z}/{x}/{y}.png",
            attribution: "Vietmap tiles served through backend proxy",
          }
        : {
            url: config.publicMapTileUrl,
            attribution: config.publicMapTileAttribution,
          },
    });
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/route") {
    await handleRoute(request, response);
    return;
  }

  if (request.method === "POST" && url.pathname === "/api/resolve-location") {
    await handleResolveLocation(request, response);
    return;
  }

  if (request.method === "GET" && url.pathname.startsWith("/api/tiles/")) {
    await handleTileProxy(url, response);
    return;
  }

  if (request.method === "GET") {
    await serveStatic(url, response);
    return;
  }

  sendText(response, 405, "Method Not Allowed");
});

server.listen(config.port, () => {
  console.log(`Server listening on http://localhost:${config.port}`);
});
