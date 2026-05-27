"use strict";

const http = require("node:http");
const fs = require("node:fs");
const fsp = require("node:fs/promises");
const path = require("node:path");
const { URL } = require("node:url");

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
};

const mimeTypes = {
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
};

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

  sendJson(response, 200, {
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
  });
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
