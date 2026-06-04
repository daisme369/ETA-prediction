"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { randomInt } = require("node:crypto");

const DEFAULT_INPUT = path.join(__dirname, "output_log.csv");
const DEFAULT_ROUTE_API = "http://localhost:3000/api/route";
const DEFAULT_MODEL_API = "http://localhost:8000/api/eta/predict";
const DEFAULT_TIME_ZONE_OFFSET = "+07:00";
const DEFAULT_RANDOM_START_DATE = "2026-04-01";
const DEFAULT_RANDOM_END_DATE = "2026-04-30";
const DEFAULT_RETRIES = 5;
const DEFAULT_RETRY_DELAY_MS = 750;

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    output: DEFAULT_INPUT,
    date: "",
    randomStartDate: DEFAULT_RANDOM_START_DATE,
    randomEndDate: DEFAULT_RANDOM_END_DATE,
    routeApi: DEFAULT_ROUTE_API,
    modelApi: DEFAULT_MODEL_API,
    vehicle: "car",
    decimals: 2,
    retries: DEFAULT_RETRIES,
    retryDelayMs: DEFAULT_RETRY_DELAY_MS,
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
    } else if (name === "--date" && value) {
      args.date = value;
      index += 1;
    } else if (name === "--random-start-date" && value) {
      args.randomStartDate = value;
      index += 1;
    } else if (name === "--random-end-date" && value) {
      args.randomEndDate = value;
      index += 1;
    } else if (name === "--route-api" && value) {
      args.routeApi = value;
      index += 1;
    } else if (name === "--model-api" && value) {
      args.modelApi = value;
      index += 1;
    } else if (name === "--vehicle" && value) {
      args.vehicle = value;
      index += 1;
    } else if (name === "--decimals" && value) {
      args.decimals = Number(value);
      index += 1;
    } else if (name === "--retries" && value) {
      args.retries = Number(value);
      index += 1;
    } else if (name === "--retry-delay-ms" && value) {
      args.retryDelayMs = Number(value);
      index += 1;
    } else if (name === "--help" || name === "-h") {
      printHelp();
      process.exit(0);
    } else if (/^\d{4}-\d{2}-\d{2}$/.test(name)) {
      args.date = name;
    } else {
      throw new Error(`Unknown or incomplete argument: ${name}`);
    }
  }

  if (args.date && !/^\d{4}-\d{2}-\d{2}$/.test(args.date)) {
    throw new Error("--date must use YYYY-MM-DD format.");
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(args.randomStartDate)) {
    throw new Error("--random-start-date must use YYYY-MM-DD format.");
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(args.randomEndDate)) {
    throw new Error("--random-end-date must use YYYY-MM-DD format.");
  }
  if (dateToUtcDay(args.randomStartDate) > dateToUtcDay(args.randomEndDate)) {
    throw new Error("--random-start-date must be before or equal to --random-end-date.");
  }
  if (!Number.isInteger(args.decimals) || args.decimals < 0 || args.decimals > 6) {
    throw new Error("--decimals must be an integer from 0 to 6.");
  }
  if (!Number.isInteger(args.retries) || args.retries < 0 || args.retries > 20) {
    throw new Error("--retries must be an integer from 0 to 20.");
  }
  if (!Number.isInteger(args.retryDelayMs) || args.retryDelayMs < 0) {
    throw new Error("--retry-delay-ms must be a non-negative integer.");
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node data/fill_output_log.js [options]

Options:
  --input <path>       CSV to read. Default: data/output_log.csv
  --output <path>      CSV to write. Default: overwrite input
  --date <YYYY-MM-DD>  Optional fixed local Vietnam date used with each row hour
  --random-start-date <YYYY-MM-DD>
                       Start of random date range. Default: ${DEFAULT_RANDOM_START_DATE}
  --random-end-date <YYYY-MM-DD>
                       End of random date range. Default: ${DEFAULT_RANDOM_END_DATE}
  --route-api <url>    Vietmap proxy endpoint. Default: ${DEFAULT_ROUTE_API}
  --model-api <url>    Model prediction endpoint. Default: ${DEFAULT_MODEL_API}
  --vehicle <name>     Vietmap vehicle. Default: car
  --decimals <n>       Decimal places for seconds. Default: 2
  --retries <n>        Retries for transient API failures. Default: ${DEFAULT_RETRIES}
  --retry-delay-ms <n> Base retry delay in milliseconds. Default: ${DEFAULT_RETRY_DELAY_MS}`);
}

function dateToUtcDay(value) {
  const [year, month, day] = value.split("-").map(Number);
  return Math.floor(Date.UTC(year, month - 1, day) / 86400000);
}

function utcDayToDate(dayNumber) {
  const date = new Date(dayNumber * 86400000);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

function randomDateBetween(startDate, endDate) {
  const startDay = dateToUtcDay(startDate);
  const endDay = dateToUtcDay(endDate);
  return utcDayToDate(startDay + randomInt(endDay - startDay + 1));
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
  return rows.slice(1).filter((values) => values.some((value) => value !== "")).map((values) => {
    const record = {};
    headers.forEach((header, index) => {
      record[header] = values[index] ?? "";
    });
    return record;
  });
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

function asFiniteNumber(record, column, rowNumber) {
  const value = Number(record[column]);
  if (!Number.isFinite(value)) {
    throw new Error(`Row ${rowNumber}: ${column} must be a finite number.`);
  }
  return value;
}

function localDepartureTime(date, hour) {
  return `${date}T${String(hour).padStart(2, "0")}:00:00${DEFAULT_TIME_ZONE_OFFSET}`;
}

function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

function isRetryableStatus(status) {
  return [408, 429, 500, 502, 503, 504].includes(status);
}

async function postJson(url, payload, args) {
  for (let attempt = 0; attempt <= args.retries; attempt += 1) {
    try {
      const response = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
        },
        body: JSON.stringify(payload),
      });

      const text = await response.text();
      let data;
      try {
        data = text ? JSON.parse(text) : {};
      } catch {
        throw new Error(`${url} returned a non-JSON response: ${text.slice(0, 160)}`);
      }

      if (!response.ok) {
        const message = data.detail || data.error || data.message || `HTTP ${response.status}`;
        const error = new Error(`${url} failed: ${Array.isArray(message) ? JSON.stringify(message) : message}`);
        error.status = response.status;
        throw error;
      }

      return data;
    } catch (error) {
      const canRetry = !error.status || isRetryableStatus(error.status);
      if (!canRetry || attempt >= args.retries) {
        throw error;
      }

      const delayMs = args.retryDelayMs * (2 ** attempt) + randomInt(0, 250);
      console.warn(`Retry ${attempt + 1}/${args.retries} for ${url} after ${delayMs}ms: ${error.message}`);
      await sleep(delayMs);
    }
  }
}

function formatSeconds(value, decimals) {
  return Number(value).toFixed(decimals);
}

async function getEstimateSeconds(args, row, rowNumber, departureDate, cache) {
  const hour = asFiniteNumber(row, "hour", rowNumber);
  if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
    throw new Error(`Row ${rowNumber}: hour must be an integer from 0 to 23.`);
  }

  const origin = {
    lat: asFiniteNumber(row, "lat", rowNumber),
    lng: asFiniteNumber(row, "lng", rowNumber),
  };
  const destination = {
    lat: asFiniteNumber(row, "destination_lat", rowNumber),
    lng: asFiniteNumber(row, "destination_lng", rowNumber),
  };
  const departureTime = localDepartureTime(departureDate, hour);
  const cacheKey = JSON.stringify({ origin, destination, hour, vehicle: args.vehicle, date: departureDate });

  if (!cache.has(cacheKey)) {
    const data = await postJson(args.routeApi, {
      origin,
      destination,
      vehicle: args.vehicle,
      departureTime,
      alternative: false,
    }, args);

    const durationMs = Number(data?.summary?.durationMs);
    if (!Number.isFinite(durationMs)) {
      throw new Error(`Row ${rowNumber}: Vietmap response did not include summary.durationMs.`);
    }
    cache.set(cacheKey, durationMs / 1000);
  }

  return cache.get(cacheKey);
}

async function getPredictSeconds(args, hour, rowNumber, cache) {
  if (!cache.has(hour)) {
    const data = await postJson(args.modelApi, { hour }, args);
    const minutes = Number(data?.prediction?.point?.minutes);
    if (!Number.isFinite(minutes)) {
      throw new Error(`Row ${rowNumber}: model response did not include prediction.point.minutes.`);
    }
    cache.set(hour, minutes * 60);
  }

  return cache.get(hour);
}

async function main() {
  const args = parseArgs(process.argv);
  const text = await fs.readFile(args.input, "utf8");
  const firstLine = text.split(/\r?\n/, 1)[0];
  const headers = firstLine.split(",");
  const records = parseCsv(text);

  for (const required of ["hour", "lat", "lng", "destination_lat", "destination_lng"]) {
    if (!headers.includes(required)) {
      throw new Error(`Missing required column: ${required}`);
    }
  }

  for (const outputColumn of ["estimate_time", "predict_time"]) {
    if (!headers.includes(outputColumn)) {
      headers.push(outputColumn);
    }
  }

  const estimateCache = new Map();
  const predictCache = new Map();

  for (let index = 0; index < records.length; index += 1) {
    const rowNumber = index + 2;
    const row = records[index];
    const hour = asFiniteNumber(row, "hour", rowNumber);
    const departureDate = args.date || randomDateBetween(args.randomStartDate, args.randomEndDate);

    const [estimateSeconds, predictSeconds] = await Promise.all([
      getEstimateSeconds(args, row, rowNumber, departureDate, estimateCache),
      getPredictSeconds(args, hour, rowNumber, predictCache),
    ]);

    row.estimate_time = formatSeconds(estimateSeconds, args.decimals);
    row.predict_time = formatSeconds(predictSeconds, args.decimals);

    if ((index + 1) % 25 === 0 || index === records.length - 1) {
      console.log(`Filled ${index + 1}/${records.length} rows`);
    }
  }

  await fs.writeFile(args.output, stringifyCsv(headers, records), "utf8");
  console.log(`Wrote ${args.output}`);
  console.log(`Vietmap calls: ${estimateCache.size}; model calls: ${predictCache.size}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
