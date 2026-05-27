"use strict";

const fs = require("node:fs");
const path = require("node:path");

const DEFAULT_NUM_ROWS = 80000;
const DEFAULT_SEED = "hanoi-eta-mock";
const DEFAULT_FORMAT = "csv";
const DEFAULT_OUTPUT_BASENAME = "mock_eta_trips";
const H3_RESOLUTION = 9;
const HANOI_LAT_RANGE = [20.9, 21.15];
const HANOI_LNG_RANGE = [105.7, 106.05];
const MOCK_START_DATE = { year: 2026, month: 1, day: 1 };
const MOCK_END_DATE = { year: 2026, month: 2, day: 28 };
const MS_PER_DAY = 24 * 60 * 60 * 1000;
const MOCK_TOTAL_DAYS =
  Math.round(
    (Date.UTC(MOCK_END_DATE.year, MOCK_END_DATE.month - 1, MOCK_END_DATE.day) -
      Date.UTC(MOCK_START_DATE.year, MOCK_START_DATE.month - 1, MOCK_START_DATE.day)) /
      MS_PER_DAY,
  ) + 1;

const LOCATIONS = [
  {
    id: "hoan_kiem_lake",
    place_name: "Hoan Kiem Lake",
    zone_name: "Hoan Kiem",
    lat: 21.0286,
    lng: 105.8522,
  },
  {
    id: "old_quarter",
    place_name: "Hanoi Old Quarter",
    zone_name: "Hoan Kiem",
    lat: 21.0359,
    lng: 105.851,
  },
  {
    id: "hanoi_railway_station",
    place_name: "Hanoi Railway Station",
    zone_name: "Dong Da",
    lat: 21.0247,
    lng: 105.8417,
  },
  {
    id: "long_bien_bridge",
    place_name: "Long Bien Bridge",
    zone_name: "Long Bien",
    lat: 21.0422,
    lng: 105.8598,
  },
  {
    id: "truc_bach_lake",
    place_name: "Truc Bach Lake",
    zone_name: "Ba Dinh",
    lat: 21.046,
    lng: 105.833,
  },
  {
    id: "lotte_center",
    place_name: "Lotte Center Hanoi",
    zone_name: "Ba Dinh",
    lat: 21.0335,
    lng: 105.814,
  },
  {
    id: "vincom_nguyen_chi_thanh",
    place_name: "Vincom Nguyen Chi Thanh",
    zone_name: "Dong Da",
    lat: 21.0238,
    lng: 105.8067,
  },
  {
    id: "indochina_plaza",
    place_name: "Indochina Plaza Hanoi",
    zone_name: "Cau Giay",
    lat: 21.0367,
    lng: 105.7828,
  },
  {
    id: "vnu_xuan_thuy",
    place_name: "Vietnam National University Hanoi",
    zone_name: "Cau Giay",
    lat: 21.0379,
    lng: 105.7818,
  },
  {
    id: "my_dinh_bus_station",
    place_name: "My Dinh Bus Station",
    zone_name: "Nam Tu Liem",
    lat: 21.0282,
    lng: 105.7787,
  },
  {
    id: "keangnam_landmark",
    place_name: "Keangnam Landmark 72",
    zone_name: "Nam Tu Liem",
    lat: 21.0167,
    lng: 105.783,
  },
  {
    id: "national_convention_center",
    place_name: "National Convention Center",
    zone_name: "Nam Tu Liem",
    lat: 21.0055,
    lng: 105.7991,
  },
  {
    id: "vincom_smart_city",
    place_name: "Vincom Mega Mall Smart City",
    zone_name: "Nam Tu Liem",
    lat: 20.9881,
    lng: 105.747,
  },
  {
    id: "royal_city",
    place_name: "Royal City",
    zone_name: "Thanh Xuan",
    lat: 21.0023,
    lng: 105.8175,
  },
  {
    id: "times_city",
    place_name: "Times City",
    zone_name: "Hai Ba Trung",
    lat: 20.9946,
    lng: 105.868,
  },
  {
    id: "bach_mai_hospital",
    place_name: "Bach Mai Hospital",
    zone_name: "Hai Ba Trung",
    lat: 20.998,
    lng: 105.846,
  },
  {
    id: "giap_bat_bus_station",
    place_name: "Giap Bat Bus Station",
    zone_name: "Hoang Mai",
    lat: 20.9719,
    lng: 105.8467,
  },
  {
    id: "aeon_long_bien",
    place_name: "Aeon Mall Long Bien",
    zone_name: "Long Bien",
    lat: 21.0109,
    lng: 105.9035,
  },
  {
    id: "gamuda_city",
    place_name: "Gamuda City",
    zone_name: "Hoang Mai",
    lat: 20.9732,
    lng: 105.8558,
  },
  {
    id: "aeon_ha_dong",
    place_name: "Aeon Mall Ha Dong",
    zone_name: "Ha Dong",
    lat: 20.999,
    lng: 105.7356,
  },
  {
    id: "utc_university",
    place_name: "University of Transport and Communications",
    zone_name: "Dong Da",
    lat: 21.0288,
    lng: 105.8005,
  },
  {
    id: "hanoi_medical_university",
    place_name: "Hanoi Medical University",
    zone_name: "Dong Da",
    lat: 21.0049,
    lng: 105.8249,
  },
  {
    id: "diplomatic_corps",
    place_name: "Diplomatic Corps Area",
    zone_name: "Bac Tu Liem",
    lat: 21.0592,
    lng: 105.8127,
  },
  {
    id: "hne_university",
    place_name: "Hanoi National University of Education",
    zone_name: "Cau Giay",
    lat: 21.0372,
    lng: 105.8037,
  },
  {
    id: "big_c_thang_long",
    place_name: "Big C Thang Long",
    zone_name: "Cau Giay",
    lat: 21.0093,
    lng: 105.7898,
  },
  {
    id: "noi_bai_airport",
    place_name: "Noi Bai Airport",
    zone_name: "Soc Son",
    lat: 21.1188,
    lng: 105.8012,
  },
];

const DAY_OF_WEEK = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
const HOUR_WEIGHTS = [
  0.15, 0.1, 0.08, 0.08, 0.1, 0.2, 0.8, 1.2, 1.1, 0.9, 0.7, 0.7, 0.8,
  0.7, 0.7, 0.7, 0.8, 1.2, 1.3, 1.1, 0.8, 0.6, 0.45, 0.25,
];
const SPEED_BY_TRAFFIC = {
  low: [10, 14],
  medium: [7, 10],
  high: [4, 7],
  severe: [2, 4],
};

let h3 = null;
try {
  h3 = require("h3-js");
} catch {
  h3 = null;
}

function printUsage() {
  console.log("Usage: node data/generate_mock_eta.js [options]");
  console.log("\nOptions:");
  console.log("  --num_rows <n>   Number of rows to generate (default 80000)");
  console.log("  --seed <value>   Seed for reproducibility (default hanoi-eta-mock)");
  console.log("  --format <type>  csv, json, or both (default csv)");
  console.log("  --output <path>  Output file for csv/json or base path for both");
  console.log("  --help           Show this message");
}

function parseArgs(argv) {
  const options = {
    numRows: DEFAULT_NUM_ROWS,
    seed: DEFAULT_SEED,
    format: DEFAULT_FORMAT,
    output: "",
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--num_rows":
      case "--num-rows":
        options.numRows = Number(argv[i + 1]);
        i += 1;
        break;
      case "--seed":
        options.seed = argv[i + 1];
        i += 1;
        break;
      case "--format":
        options.format = String(argv[i + 1] || "").toLowerCase();
        i += 1;
        break;
      case "--output":
        options.output = String(argv[i + 1] || "");
        i += 1;
        break;
      case "--help":
        printUsage();
        process.exit(0);
        break;
      default:
        if (arg.startsWith("--")) {
          console.error(`Unknown option: ${arg}`);
          printUsage();
          process.exit(1);
        }
    }
  }

  if (!Number.isInteger(options.numRows) || options.numRows <= 0) {
    throw new Error("--num_rows must be a positive integer.");
  }

  if (!["csv", "json", "both"].includes(options.format)) {
    throw new Error("--format must be csv, json, or both.");
  }

  return options;
}

function xmur3(str) {
  let h = 1779033703 ^ str.length;
  for (let i = 0; i < str.length; i += 1) {
    h = Math.imul(h ^ str.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  return function () {
    h = Math.imul(h ^ (h >>> 16), 2246822507);
    h = Math.imul(h ^ (h >>> 13), 3266489909);
    h ^= h >>> 16;
    return h >>> 0;
  };
}

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a += 0x6d2b79f5;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function createRng(seedInput) {
  const numericSeed = Number(seedInput);
  const seed = Number.isFinite(numericSeed)
    ? numericSeed
    : xmur3(String(seedInput))();
  return mulberry32(seed);
}

function randomFloat(rng, min = 0, max = 1) {
  return min + (max - min) * rng();
}

function randomInt(rng, min, max) {
  return Math.floor(randomFloat(rng, min, max + 1));
}

function weightedChoice(rng, options) {
  const total = options.reduce((sum, option) => sum + option.weight, 0);
  const target = randomFloat(rng, 0, total);
  let cumulative = 0;
  for (const option of options) {
    cumulative += option.weight;
    if (target <= cumulative) {
      return option.value;
    }
  }
  return options[options.length - 1].value;
}

function roundTo(value, decimals) {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function pad2(value) {
  return String(value).padStart(2, "0");
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

function withinRange(value, range) {
  return value >= range[0] && value <= range[1];
}

function computeH3Id(location) {
  if (h3 && typeof h3.latLngToCell === "function") {
    return h3.latLngToCell(location.lat, location.lng, H3_RESOLUTION);
  }
  // Deterministic fallback so the same location always maps to the same H3-like bucket.
  return `h3_${location.id}`;
}

function sampleHourOfDay(rng) {
  const options = HOUR_WEIGHTS.map((weight, hour) => ({ value: hour, weight }));
  return weightedChoice(rng, options);
}

function dayNameFromDate(date) {
  const jsDay = date.getUTCDay();
  return ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"][jsDay];
}

function sampleTripTimestamp(rng) {
  const dayOffset = randomInt(rng, 0, MOCK_TOTAL_DAYS - 1);
  const hour = sampleHourOfDay(rng);
  const minute = randomInt(rng, 0, 59);
  const second = randomInt(rng, 0, 59);
  const utcDate = new Date(
    Date.UTC(
      MOCK_START_DATE.year,
      MOCK_START_DATE.month - 1,
      MOCK_START_DATE.day + dayOffset,
      hour,
      minute,
      second,
    ),
  );

  const year = utcDate.getUTCFullYear();
  const month = utcDate.getUTCMonth() + 1;
  const day = utcDate.getUTCDate();
  const timestamp = `${year}-${pad2(month)}-${pad2(day)}T${pad2(hour)}:${pad2(
    minute,
  )}:${pad2(second)}+07:00`;

  return {
    timestamp,
    tripDate: `${year}-${pad2(month)}-${pad2(day)}`,
    month,
    dayOfMonth: day,
    dayOfWeek: dayNameFromDate(utcDate),
    hourOfDay: hour,
    minuteOfHour: minute,
  };
}

function sampleTrafficLevel(rng, hour, isRushHour) {
  if (isRushHour) {
    return weightedChoice(rng, [
      { value: "low", weight: 0.05 },
      { value: "medium", weight: 0.25 },
      { value: "high", weight: 0.45 },
      { value: "severe", weight: 0.25 },
    ]);
  }

  if (hour <= 5) {
    return weightedChoice(rng, [
      { value: "low", weight: 0.6 },
      { value: "medium", weight: 0.3 },
      { value: "high", weight: 0.1 },
      { value: "severe", weight: 0.0 },
    ]);
  }

  if (hour >= 21) {
    return weightedChoice(rng, [
      { value: "low", weight: 0.45 },
      { value: "medium", weight: 0.4 },
      { value: "high", weight: 0.15 },
      { value: "severe", weight: 0.0 },
    ]);
  }

  return weightedChoice(rng, [
    { value: "low", weight: 0.35 },
    { value: "medium", weight: 0.4 },
    { value: "high", weight: 0.2 },
    { value: "severe", weight: 0.05 },
  ]);
}

function sampleRainLevel(rng, isRaining) {
  if (!isRaining) {
    return "none";
  }
  return weightedChoice(rng, [
    { value: "light", weight: 0.4 },
    { value: "moderate", weight: 0.3 },
    { value: "heavy", weight: 0.2 },
    { value: "very_heavy", weight: 0.1 },
  ]);
}

function sampleWeatherCondition(rng, isRaining, rainLevel) {
  if (isRaining) {
    if (rainLevel === "heavy" || rainLevel === "very_heavy") {
      return weightedChoice(rng, [
        { value: "storm", weight: 0.6 },
        { value: "rain", weight: 0.4 },
      ]);
    }
    return "rain";
  }
  return weightedChoice(rng, [
    { value: "clear", weight: 0.5 },
    { value: "cloudy", weight: 0.4 },
    { value: "fog", weight: 0.1 },
  ]);
}

function sampleDetourFactor(rng, haversineMetersValue) {
  // Short trips should not have extreme detours.
  if (haversineMetersValue < 1500) {
    return randomFloat(rng, 1.05, 1.25);
  }
  if (haversineMetersValue < 5000) {
    return randomFloat(rng, 1.08, 1.4);
  }
  return randomFloat(rng, 1.1, 1.8);
}

function computeBaselineEtaSeconds(rng, distanceMeters, trafficLevel) {
  const [minSpeed, maxSpeed] = SPEED_BY_TRAFFIC[trafficLevel];

  for (let attempt = 0; attempt < 8; attempt += 1) {
    // Baseline ETA should reflect traffic-driven speed bands.
    const speed = randomFloat(rng, minSpeed, maxSpeed);
    const eta = Math.round(distanceMeters / speed);
    if (eta >= 120 && eta <= 7200) {
      return eta;
    }
  }

  const fallbackSpeed = (minSpeed + maxSpeed) / 2;
  const fallbackEta = Math.round(distanceMeters / fallbackSpeed);
  return Math.max(120, Math.min(7200, fallbackEta));
}

function computeResidualSeconds(rng, context) {
  let residual = randomInt(rng, -180, 180); // Base noise for day-to-day variance.

  if (context.isRushHour) {
    residual += randomInt(rng, 120, 600);
  }

  if (context.trafficLevel === "high") {
    residual += randomInt(rng, 180, 600);
  }

  if (context.trafficLevel === "severe") {
    residual += randomInt(rng, 300, 1200);
  }

  if (context.rainLevel === "light") {
    residual += randomInt(rng, 60, 180);
  } else if (context.rainLevel === "moderate") {
    residual += randomInt(rng, 120, 300);
  } else if (context.rainLevel === "heavy") {
    residual += randomInt(rng, 300, 700);
  } else if (context.rainLevel === "very_heavy") {
    residual += randomInt(rng, 500, 1000);
  }

  if (context.isHoliday) {
    // Holidays can reduce congestion (negative) or create surge delays (positive).
    residual += randomInt(rng, -120, 600);
  }

  if (!context.isRushHour && context.trafficLevel === "low" && !context.isRaining) {
    // Allow some negative residuals when traffic is unusually smooth.
    residual += randomInt(rng, -240, 0);
  }

  return Math.max(-900, Math.min(1800, residual));
}

function buildRow(rng, locations) {
  const origin = weightedChoice(rng, locations.map((location) => ({ value: location, weight: 1 })));
  let destination = weightedChoice(rng, locations.map((location) => ({ value: location, weight: 1 })));

  // Prefer cross-zone trips without forbidding same-zone rides entirely.
  let attempts = 0;
  while (
    (destination.id === origin.id ||
      (destination.zone_name === origin.zone_name && rng() < 0.8)) &&
    attempts < 8
  ) {
    destination = weightedChoice(rng, locations.map((location) => ({ value: location, weight: 1 })));
    attempts += 1;
  }

  if (destination.id === origin.id) {
    destination = locations.find((location) => location.id !== origin.id) || destination;
  }

  const tripTime = sampleTripTimestamp(rng);
  const hourOfDay = tripTime.hourOfDay;
  const isRushHour = [7, 8, 17, 18, 19].includes(hourOfDay);
  const dayOfWeek = tripTime.dayOfWeek;
  const isWeekend = dayOfWeek === "Sat" || dayOfWeek === "Sun";
  const isHoliday = rng() < 0.07; // Low holiday probability for realism.

  const isRaining = rng() < 0.22; // A moderate chance to show weather impact in the MVP.
  const rainLevel = sampleRainLevel(rng, isRaining);
  const weatherCondition = sampleWeatherCondition(rng, isRaining, rainLevel);

  const trafficLevel = sampleTrafficLevel(rng, hourOfDay, isRushHour);

  const haversineMetersValue = haversineMeters(
    origin.lat,
    origin.lng,
    destination.lat,
    destination.lng,
  );
  const roundedHaversine = roundTo(haversineMetersValue, 1);

  const detourFactor = sampleDetourFactor(rng, roundedHaversine);
  let baselineDistance = roundTo(roundedHaversine * detourFactor, 1);
  baselineDistance = Math.max(baselineDistance, roundedHaversine);

  const baselineEtaSeconds = computeBaselineEtaSeconds(rng, baselineDistance, trafficLevel);

  const residualSeconds = computeResidualSeconds(rng, {
    isRushHour,
    trafficLevel,
    rainLevel,
    isHoliday,
    isRaining,
  });

  let actualEtaSeconds = baselineEtaSeconds + residualSeconds;
  if (actualEtaSeconds < 60) {
    actualEtaSeconds = 60;
  }

  const adjustedResidualSeconds = actualEtaSeconds - baselineEtaSeconds;

  return {
    trip_timestamp: tripTime.timestamp,
    trip_date: tripTime.tripDate,
    month: tripTime.month,
    day_of_month: tripTime.dayOfMonth,
    minute_of_hour: tripTime.minuteOfHour,
    origin_h3: origin.h3_id,
    destination_h3: destination.h3_id,
    origin_lng: origin.lng,
    origin_lat: origin.lat,
    destination_lng: destination.lng,
    destination_lat: destination.lat,
    hour_of_day: hourOfDay,
    is_rush_hour: isRushHour,
    day_of_week: dayOfWeek,
    is_weekend: isWeekend,
    is_holiday: isHoliday,
    haversine_distance_meters: roundedHaversine,
    baseline_distance_meters: baselineDistance,
    traffic_level: trafficLevel,
    is_raining: isRaining,
    rain_level: rainLevel,
    weather_condition: weatherCondition,
    baseline_eta_secs: baselineEtaSeconds,
    actual_eta_secs: actualEtaSeconds,
    residual_secs: adjustedResidualSeconds,
  };
}

function validateRow(row, index) {
  const prefix = `Row ${index + 1}:`;

  if (!/^2026-(01|02)-\d{2}T\d{2}:\d{2}:\d{2}\+07:00$/.test(row.trip_timestamp)) {
    throw new Error(`${prefix} trip_timestamp must be in the 2026-01..2026-02 Hanoi window.`);
  }

  const timestampDate = row.trip_timestamp.slice(0, 10);
  if (timestampDate < "2026-01-01" || timestampDate > "2026-02-28") {
    throw new Error(`${prefix} trip_timestamp outside 2026-01-01..2026-02-28.`);
  }

  if (row.trip_date !== timestampDate) {
    throw new Error(`${prefix} trip_date inconsistent with trip_timestamp.`);
  }

  if (row.month !== Number(timestampDate.slice(5, 7))) {
    throw new Error(`${prefix} month inconsistent with trip_timestamp.`);
  }

  if (row.day_of_month !== Number(timestampDate.slice(8, 10))) {
    throw new Error(`${prefix} day_of_month inconsistent with trip_timestamp.`);
  }

  if (row.hour_of_day !== Number(row.trip_timestamp.slice(11, 13))) {
    throw new Error(`${prefix} hour_of_day inconsistent with trip_timestamp.`);
  }

  if (row.minute_of_hour !== Number(row.trip_timestamp.slice(14, 16))) {
    throw new Error(`${prefix} minute_of_hour inconsistent with trip_timestamp.`);
  }

  if (!withinRange(row.origin_lat, HANOI_LAT_RANGE)) {
    throw new Error(`${prefix} origin_lat outside Hanoi range.`);
  }
  if (!withinRange(row.origin_lng, HANOI_LNG_RANGE)) {
    throw new Error(`${prefix} origin_lng outside Hanoi range.`);
  }
  if (!withinRange(row.destination_lat, HANOI_LAT_RANGE)) {
    throw new Error(`${prefix} destination_lat outside Hanoi range.`);
  }
  if (!withinRange(row.destination_lng, HANOI_LNG_RANGE)) {
    throw new Error(`${prefix} destination_lng outside Hanoi range.`);
  }

  // Detect lat/lng swap mistakes explicitly for clear validation output.
  if (withinRange(row.origin_lat, HANOI_LNG_RANGE) && withinRange(row.origin_lng, HANOI_LAT_RANGE)) {
    throw new Error(`${prefix} origin lat/lng appear swapped.`);
  }
  if (
    withinRange(row.destination_lat, HANOI_LNG_RANGE) &&
    withinRange(row.destination_lng, HANOI_LAT_RANGE)
  ) {
    throw new Error(`${prefix} destination lat/lng appear swapped.`);
  }

  if (
    Math.abs(row.origin_lat - row.destination_lat) < 1e-6 &&
    Math.abs(row.origin_lng - row.destination_lng) < 1e-6
  ) {
    throw new Error(`${prefix} origin and destination should not match.`);
  }

  if (row.baseline_distance_meters + 0.01 < row.haversine_distance_meters) {
    throw new Error(`${prefix} baseline_distance_meters must be >= haversine_distance_meters.`);
  }

  if (row.is_weekend !== (row.day_of_week === "Sat" || row.day_of_week === "Sun")) {
    throw new Error(`${prefix} is_weekend inconsistent with day_of_week.`);
  }

  if (row.is_rush_hour !== [7, 8, 17, 18, 19].includes(row.hour_of_day)) {
    throw new Error(`${prefix} is_rush_hour inconsistent with hour_of_day.`);
  }

  if (!row.is_raining && row.rain_level !== "none") {
    throw new Error(`${prefix} rain_level must be none when is_raining is false.`);
  }

  if (row.is_raining && row.rain_level === "none") {
    throw new Error(`${prefix} rain_level must be set when is_raining is true.`);
  }

  if (row.is_raining && !["rain", "storm"].includes(row.weather_condition)) {
    throw new Error(`${prefix} weather_condition must be rain or storm when raining.`);
  }

  if (!row.is_raining && ["rain", "storm"].includes(row.weather_condition)) {
    throw new Error(`${prefix} weather_condition must be clear/cloudy/fog when not raining.`);
  }

  if (row.actual_eta_secs !== row.baseline_eta_secs + row.residual_secs) {
    throw new Error(`${prefix} actual_eta_secs must equal baseline_eta_secs + residual_secs.`);
  }

  if (row.residual_secs !== row.actual_eta_secs - row.baseline_eta_secs) {
    throw new Error(`${prefix} residual_secs must equal actual_eta_secs - baseline_eta_secs.`);
  }

  if (row.actual_eta_secs < 60) {
    throw new Error(`${prefix} actual_eta_secs must be >= 60.`);
  }
}

function escapeCsv(value) {
  const raw = value === null || value === undefined ? "" : String(value);
  if (/[,\"\n]/.test(raw)) {
    return `"${raw.replace(/\"/g, "\"\"")}"`;
  }
  return raw;
}

function toCsv(rows) {
  const headers = [
    "trip_timestamp",
    "trip_date",
    "month",
    "day_of_month",
    "minute_of_hour",
    "origin_h3",
    "destination_h3",
    "origin_lng",
    "origin_lat",
    "destination_lng",
    "destination_lat",
    "hour_of_day",
    "is_rush_hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "haversine_distance_meters",
    "baseline_distance_meters",
    "traffic_level",
    "is_raining",
    "rain_level",
    "weather_condition",
    "baseline_eta_secs",
    "actual_eta_secs",
    "residual_secs",
  ];

  const lines = [headers.join(",")];
  for (const row of rows) {
    const values = headers.map((header) => escapeCsv(row[header]));
    lines.push(values.join(","));
  }
  return lines.join("\n");
}

function resolveOutputPaths(options) {
  const scriptDir = __dirname;

  if (options.format === "both") {
    const base = options.output
      ? stripExtension(options.output)
      : path.join(scriptDir, DEFAULT_OUTPUT_BASENAME);
    return {
      csvPath: `${base}.csv`,
      jsonPath: `${base}.json`,
    };
  }

  if (options.format === "csv") {
    const outputPath = options.output || path.join(scriptDir, `${DEFAULT_OUTPUT_BASENAME}.csv`);
    return { csvPath: outputPath, jsonPath: "" };
  }

  const outputPath = options.output || path.join(scriptDir, `${DEFAULT_OUTPUT_BASENAME}.json`);
  return { csvPath: "", jsonPath: outputPath };
}

function stripExtension(filePath) {
  const ext = path.extname(filePath);
  if (!ext) {
    return filePath;
  }
  return filePath.slice(0, -ext.length);
}

function ensureDir(filePath) {
  const dir = path.dirname(filePath);
  fs.mkdirSync(dir, { recursive: true });
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  const rng = createRng(options.seed);

  const locations = LOCATIONS.map((location) => ({
    ...location,
    h3_id: computeH3Id(location),
  }));

  const rows = [];
  for (let i = 0; i < options.numRows; i += 1) {
    const row = buildRow(rng, locations);
    validateRow(row, i);
    rows.push(row);
  }

  const { csvPath, jsonPath } = resolveOutputPaths(options);
  if (csvPath) {
    ensureDir(csvPath);
    fs.writeFileSync(csvPath, toCsv(rows), "utf8");
  }
  if (jsonPath) {
    ensureDir(jsonPath);
    fs.writeFileSync(jsonPath, JSON.stringify(rows, null, 2), "utf8");
  }

  const outputs = [csvPath, jsonPath].filter(Boolean).map((file) => path.relative(process.cwd(), file));
  console.log(`Generated ${rows.length} rows.`);
  console.log(`Output: ${outputs.join(", ")}`);
}

main();
