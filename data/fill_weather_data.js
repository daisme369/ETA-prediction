"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");

const DEFAULT_INPUT = path.join(__dirname, "processed_data.csv");
const DEFAULT_WEATHER_API = "https://archive-api.open-meteo.com/v1/archive";
const DEFAULT_TIMEZONE = "Asia/Bangkok";
const DEFAULT_HOURLY = "rain";
const DEFAULT_DECIMALS = 3;

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    output: DEFAULT_INPUT,
    weatherApi: DEFAULT_WEATHER_API,
    timezone: DEFAULT_TIMEZONE,
    hourly: DEFAULT_HOURLY,
    decimals: DEFAULT_DECIMALS,
  };

  for (let index = 2; index < argv.length; index += 1) {
    const name = argv[index];
    const value = argv[index + 1];

    if (name === "--input" && value) {
      args.input = path.resolve(value);
      index += 1;
    } else if (name === "--output" && value) {
      args.output = path.resolve(value);
      index += 1;
    } else if (name === "--weather-api" && value) {
      args.weatherApi = value;
      index += 1;
    } else if (name === "--timezone" && value) {
      args.timezone = value;
      index += 1;
    } else if (name === "--hourly" && value) {
      args.hourly = value;
      index += 1;
    } else if (name === "--decimals" && value) {
      args.decimals = Number(value);
      index += 1;
    } else if (name === "--help" || name === "-h") {
      printHelp();
      process.exit(0);
    } else {
      throw new Error(`Unknown or incomplete argument: ${name}`);
    }
  }

  if (!Number.isInteger(args.decimals) || args.decimals < 0 || args.decimals > 6) {
    throw new Error("--decimals must be an integer from 0 to 6.");
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node data/fill_weather_data.js [options]

Options:
  --input <path>        CSV to read. Default: data/processed_data.csv
  --output <path>       CSV to write. Default: overwrite input
  --weather-api <url>   Open-Meteo archive endpoint. Default: ${DEFAULT_WEATHER_API}
  --timezone <name>     API timezone. Default: ${DEFAULT_TIMEZONE}
  --hourly <name>       Hourly weather variable. Default: ${DEFAULT_HOURLY}
  --decimals <n>        Decimal places for numeric weather values. Default: ${DEFAULT_DECIMALS}`);
}

function parseCsv(text) {
  const rows = [];
  let field = "";
  let row = [];
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];

    if (quoted) {
      if (char === "\"" && next === "\"") {
        field += "\"";
        index += 1;
      } else if (char === "\"") {
        quoted = false;
      } else {
        field += char;
      }
      continue;
    }

    if (char === "\"") {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else if (char !== "\r") {
      field += char;
    }
  }

  if (field || row.length) {
    row.push(field);
    rows.push(row);
  }

  if (!rows.length) {
    throw new Error("CSV is empty.");
  }

  const headers = rows[0];
  const records = rows.slice(1).filter((values) => values.some((value) => value !== "")).map((values) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = values[index] ?? "";
    });
    return record;
  });

  return { headers, records };
}

function stringifyCsv(headers, records) {
  return [
    headers.join(","),
    ...records.map((record) => headers.map((header) => escapeCsv(record[header] ?? "")).join(",")),
  ].join("\n") + "\n";
}

function escapeCsv(value) {
  const text = String(value);
  if (/[",\r\n]/.test(text)) {
    return `"${text.replace(/"/g, "\"\"")}"`;
  }
  return text;
}

function requireColumns(headers, columns) {
  for (const column of columns) {
    if (!headers.includes(column)) {
      throw new Error(`Missing required column: ${column}`);
    }
  }
}

function addColumns(headers, columns) {
  for (const column of columns) {
    if (!headers.includes(column)) {
      headers.push(column);
    }
  }
}

function minMaxDates(records) {
  const dates = records.map((record) => record.date).filter(Boolean).sort();
  if (!dates.length) {
    throw new Error("No date values found.");
  }
  return { startDate: dates[0], endDate: dates[dates.length - 1] };
}

function asFiniteNumber(record, column, rowNumber) {
  const value = Number(record[column]);
  if (!Number.isFinite(value)) {
    throw new Error(`Row ${rowNumber}: ${column} must be a finite number.`);
  }
  return value;
}

function normalizeHour(value, rowNumber) {
  const hour = Number(value);
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
    throw new Error(`Row ${rowNumber}: hour must be an integer from 0 to 23.`);
  }
  return String(hour).padStart(2, "0");
}

function coordinateKey(lat, lng) {
  return `${Number(lat).toFixed(6)},${Number(lng).toFixed(6)}`;
}

function weatherTimeKey(date, hour) {
  return `${date}T${String(hour).padStart(2, "0")}:00`;
}

async function fetchWeather(args, coordinates, startDate, endDate) {
  const url = new URL(args.weatherApi);
  url.searchParams.set("latitude", coordinates.map((coordinate) => coordinate.lat).join(","));
  url.searchParams.set("longitude", coordinates.map((coordinate) => coordinate.lng).join(","));
  url.searchParams.set("start_date", startDate);
  url.searchParams.set("end_date", endDate);
  url.searchParams.set("hourly", args.hourly);
  url.searchParams.set("timezone", args.timezone);

  const response = await fetch(url);
  const text = await response.text();

  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`Weather API returned a non-JSON response: ${text.slice(0, 160)}`);
  }

  if (!response.ok) {
    const message = data.reason || data.error || data.message || `HTTP ${response.status}`;
    throw new Error(`Weather API failed: ${message}`);
  }

  return Array.isArray(data) ? data : [data];
}

function buildWeatherLookup(apiResponses, coordinates, hourlyVariable) {
  const lookup = new Map();

  apiResponses.forEach((response, index) => {
    const coordinate = coordinates[index];
    if (!coordinate) {
      return;
    }

    const times = response?.hourly?.time;
    const values = response?.hourly?.[hourlyVariable];
    if (!Array.isArray(times) || !Array.isArray(values)) {
      throw new Error(`Weather response for coordinate ${index + 1} did not include hourly.${hourlyVariable}.`);
    }

    const byTime = new Map();
    times.forEach((time, timeIndex) => {
      byTime.set(time, values[timeIndex]);
    });
    lookup.set(coordinateKey(coordinate.lat, coordinate.lng), byTime);
  });

  return lookup;
}

function formatWeatherValue(value, decimals) {
  if (value === null || value === undefined || value === "") {
    return "";
  }

  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric.toFixed(decimals) : String(value);
}

async function main() {
  const args = parseArgs(process.argv);
  const text = await fs.readFile(args.input, "utf8");
  const { headers, records } = parseCsv(text);

  requireColumns(headers, ["date", "hour", "lat", "lng", "destination_lat", "destination_lng"]);
  addColumns(headers, ["origin_rain", "destination_rain", "rain"]);

  const { startDate, endDate } = minMaxDates(records);
  const first = records[0];
  const coordinates = [
    {
      label: "origin",
      lat: asFiniteNumber(first, "lat", 2),
      lng: asFiniteNumber(first, "lng", 2),
    },
    {
      label: "destination",
      lat: asFiniteNumber(first, "destination_lat", 2),
      lng: asFiniteNumber(first, "destination_lng", 2),
    },
  ];

  const apiResponses = await fetchWeather(args, coordinates, startDate, endDate);
  const weatherLookup = buildWeatherLookup(apiResponses, coordinates, args.hourly);

  let missing = 0;
  records.forEach((record, index) => {
    const rowNumber = index + 2;
    const hour = normalizeHour(record.hour, rowNumber);
    const timeKey = weatherTimeKey(record.date, hour);
    const originKey = coordinateKey(record.lat, record.lng);
    const destinationKey = coordinateKey(record.destination_lat, record.destination_lng);

    const originValue = weatherLookup.get(originKey)?.get(timeKey);
    const destinationValue = weatherLookup.get(destinationKey)?.get(timeKey);

    if (originValue === undefined || destinationValue === undefined) {
      missing += 1;
    }

    record.origin_rain = formatWeatherValue(originValue, args.decimals);
    record.destination_rain = formatWeatherValue(destinationValue, args.decimals);
    record.rain = record.destination_rain;
  });

  await fs.writeFile(args.output, stringifyCsv(headers, records), "utf8");
  console.log(`Fetched ${args.hourly} for ${coordinates.length} coordinates from ${startDate} to ${endDate}.`);
  console.log(`Wrote ${args.output}`);
  console.log(`Rows: ${records.length}; missing weather rows: ${missing}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
