"use strict";

const fs = require("node:fs/promises");
const path = require("node:path");
const { randomInt } = require("node:crypto");

const DEFAULT_INPUT = path.join(__dirname, "output_log.csv");
const DEFAULT_TIMESTAMP_SOURCE = path.join(__dirname, "processed_data.csv");
const DEFAULT_ROUTE_API = "http://localhost:3000/api/route";
const DEFAULT_MODEL_API = "http://localhost:8000/api/eta/predict";
const DEFAULT_TIME_ZONE_OFFSET = "+07:00";
const DEFAULT_RETRIES = 5;
const DEFAULT_RETRY_DELAY_MS = 750;
const DEFAULT_ROUTE_DELAY_MS = 300;
const DEFAULT_SAVE_EVERY = 25;

function parseArgs(argv) {
  const args = {
    input: DEFAULT_INPUT,
    output: DEFAULT_INPUT,
    timestampSource: DEFAULT_TIMESTAMP_SOURCE,
    routeApi: DEFAULT_ROUTE_API,
    modelApi: DEFAULT_MODEL_API,
    vehicle: "car",
    decimals: 2,
    retries: DEFAULT_RETRIES,
    retryDelayMs: DEFAULT_RETRY_DELAY_MS,
    routeDelayMs: DEFAULT_ROUTE_DELAY_MS,
    saveEvery: DEFAULT_SAVE_EVERY,
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
    } else if (name === "--timestamp-source" && value) {
      args.timestampSource = path.resolve(value);
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
    } else if (name === "--route-delay-ms" && value) {
      args.routeDelayMs = Number(value);
      index += 1;
    } else if (name === "--save-every" && value) {
      args.saveEvery = Number(value);
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
  if (!Number.isInteger(args.retries) || args.retries < 0 || args.retries > 20) {
    throw new Error("--retries must be an integer from 0 to 20.");
  }
  if (!Number.isInteger(args.retryDelayMs) || args.retryDelayMs < 0) {
    throw new Error("--retry-delay-ms must be a non-negative integer.");
  }
  if (!Number.isInteger(args.routeDelayMs) || args.routeDelayMs < 0) {
    throw new Error("--route-delay-ms must be a non-negative integer.");
  }
  if (!Number.isInteger(args.saveEvery) || args.saveEvery < 1) {
    throw new Error("--save-every must be a positive integer.");
  }

  return args;
}

function printHelp() {
  console.log(`Usage:
  node data/fill_output_log.js [options]

Options:
  --input <path>             CSV to read. Default: data/output_log.csv
  --output <path>            CSV to write. Default: overwrite input
  --timestamp-source <path>  CSV source for timestamp/hour alignment. Default: data/processed_data.csv
  --route-api <url>          Vietmap proxy endpoint. Default: ${DEFAULT_ROUTE_API}
  --model-api <url>          Model prediction endpoint. Default: ${DEFAULT_MODEL_API}
  --vehicle <name>           Vietmap vehicle. Default: car
  --decimals <n>             Decimal places for seconds. Default: 2
  --retries <n>              Retries for transient API failures. Default: ${DEFAULT_RETRIES}
  --retry-delay-ms <n>       Base retry delay in milliseconds. Default: ${DEFAULT_RETRY_DELAY_MS}
  --route-delay-ms <n>       Delay after new Vietmap calls. Default: ${DEFAULT_ROUTE_DELAY_MS}
  --save-every <n>           Persist CSV progress every n rows. Default: ${DEFAULT_SAVE_EVERY}`);
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

  throw new Error(`${url} failed without returning a response.`);
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
  return hour;
}

function unixSecondsToTimestamp(value) {
  const seconds = Number(value);
  if (!Number.isFinite(seconds)) {
    return "";
  }

  const date = new Date(seconds * 1000);
  const year = date.getUTCFullYear();
  const month = String(date.getUTCMonth() + 1).padStart(2, "0");
  const day = String(date.getUTCDate()).padStart(2, "0");
  const hour = String(date.getUTCHours()).padStart(2, "0");
  const minute = String(date.getUTCMinutes()).padStart(2, "0");
  const second = String(date.getUTCSeconds()).padStart(2, "0");
  return `${year}-${month}-${day} ${hour}:${minute}:${second}`;
}

function sourceTimestamp(record) {
  if (record.timestamp) {
    return record.timestamp.trim();
  }
  if (record.datetime) {
    return unixSecondsToTimestamp(record.datetime);
  }
  if (record.date && record.time) {
    return `${record.date.trim()} ${record.time.trim()}`;
  }
  return "";
}

function timestampHour(timestamp) {
  const match = /^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}):(\d{2})(?::(\d{2}))?/.exec(timestamp);
  if (!match) {
    throw new Error(`Invalid timestamp format: ${timestamp}`);
  }
  return Number(match[4]);
}

function departureTimeFromTimestamp(timestamp) {
  const match = /^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}(?::\d{2})?)/.exec(timestamp);
  if (!match) {
    throw new Error(`Invalid timestamp format: ${timestamp}`);
  }

  const time = match[2].length === 5 ? `${match[2]}:00` : match[2];
  return `${match[1]}T${time}${DEFAULT_TIME_ZONE_OFFSET}`;
}

function sameDelta(left, right) {
  if (!left || !right) {
    return false;
  }
  return Math.abs(Number(left) - Number(right)) < 0.000001;
}

function buildSourceByDelta(sourceRecords) {
  const sourceByDelta = new Map();
  const duplicateDeltas = new Set();

  sourceRecords.forEach((record) => {
    if (!record.delta_time) {
      return;
    }
    if (sourceByDelta.has(record.delta_time)) {
      duplicateDeltas.add(record.delta_time);
      return;
    }
    sourceByDelta.set(record.delta_time, record);
  });

  for (const delta of duplicateDeltas) {
    sourceByDelta.delete(delta);
  }

  return sourceByDelta;
}

function hydrateTimestamps(records, sourceRecords) {
  if (!sourceRecords.length && records.every((record) => record.timestamp)) {
    return { records, droppedRows: 0 };
  }

  const sourceByDelta = buildSourceByDelta(sourceRecords);
  const hydrated = [];
  let droppedRows = 0;

  records.forEach((record, index) => {
    let source = sourceRecords[index];
    if (source && record.delta_time && source.delta_time && !sameDelta(record.delta_time, source.delta_time)) {
      source = sourceByDelta.get(record.delta_time);
    }

    if (!source && !record.timestamp) {
      droppedRows += 1;
      return;
    }

    const timestamp = source ? sourceTimestamp(source) : record.timestamp;
    if (!timestamp) {
      droppedRows += 1;
      return;
    }

    record.timestamp = timestamp;
    const hourFromTimestamp = timestampHour(timestamp);
    const sourceHour = source?.hour ? normalizeHour(source.hour, index + 2) : hourFromTimestamp;

    if (sourceHour !== hourFromTimestamp) {
      throw new Error(`Row ${index + 2}: source hour ${sourceHour} does not match timestamp ${timestamp}.`);
    }

    record.hour = String(hourFromTimestamp);
    hydrated.push(record);
  });

  return { records: hydrated, droppedRows };
}

function formatSeconds(value, decimals) {
  return Number(value).toFixed(decimals);
}

function extractPredictionSeconds(data, rowNumber) {
  const secondCandidates = [
    data?.prediction?.point?.seconds,
    data?.prediction?.seconds,
    data?.point?.seconds,
    data?.seconds,
    data?.predict_time,
    data?.predicted_time,
    data?.eta_seconds,
  ];

  for (const value of secondCandidates) {
    const seconds = Number(value);
    if (Number.isFinite(seconds)) {
      return seconds;
    }
  }

  const minuteCandidates = [
    data?.prediction?.point?.minutes,
    data?.prediction?.minutes,
    data?.point?.minutes,
    data?.minutes,
    data?.eta_minutes,
  ];

  for (const value of minuteCandidates) {
    const minutes = Number(value);
    if (Number.isFinite(minutes)) {
      return minutes * 60;
    }
  }

  throw new Error(`Row ${rowNumber}: model response did not include a supported prediction value.`);
}

async function getEstimateSeconds(args, row, rowNumber, departureTime, cache) {
  const origin = {
    lat: asFiniteNumber(row, "lat", rowNumber),
    lng: asFiniteNumber(row, "lng", rowNumber),
  };
  const destination = {
    lat: asFiniteNumber(row, "destination_lat", rowNumber),
    lng: asFiniteNumber(row, "destination_lng", rowNumber),
  };
  const cacheKey = JSON.stringify({ origin, destination, vehicle: args.vehicle, departureTime });

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
    if (args.routeDelayMs > 0) {
      await sleep(args.routeDelayMs);
    }
  }

  return cache.get(cacheKey);
}

async function getPredictSeconds(args, row, rowNumber, departureTime, cache) {
  const hour = normalizeHour(row.hour, rowNumber);
  const cacheKey = departureTime;

  if (!cache.has(cacheKey)) {
    const data = await postJson(args.modelApi, {
      departure_time: departureTime,
      hour,
    }, args);
    cache.set(cacheKey, extractPredictionSeconds(data, rowNumber));
  }

  return cache.get(cacheKey);
}

async function readTimestampSource(args) {
  try {
    const text = await fs.readFile(args.timestampSource, "utf8");
    return parseCsv(text).records;
  } catch (error) {
    if (error.code === "ENOENT") {
      return [];
    }
    throw error;
  }
}

function hasFilledTimes(row) {
  return row.estimate_time !== undefined
    && row.predict_time !== undefined
    && String(row.estimate_time).trim() !== ""
    && String(row.predict_time).trim() !== "";
}

async function writeOutput(args, headers, records) {
  await fs.writeFile(args.output, stringifyCsv(headers, records), "utf8");
}

async function main() {
  const args = parseArgs(process.argv);
  const text = await fs.readFile(args.input, "utf8");
  const { headers, records: rawRecords } = parseCsv(text);
  const sourceRecords = await readTimestampSource(args);

  requireColumns(headers, ["hour", "lat", "lng", "destination_lat", "destination_lng"]);
  addColumns(headers, ["timestamp", "estimate_time", "predict_time"]);

  const { records, droppedRows } = hydrateTimestamps(rawRecords, sourceRecords);
  if (!records.length) {
    throw new Error("No rows with usable timestamp data were found.");
  }

  const estimateCache = new Map();
  const predictCache = new Map();

  for (let index = 0; index < records.length; index += 1) {
    const rowNumber = index + 2;
    const row = records[index];
    const departureTime = departureTimeFromTimestamp(row.timestamp);

    if (hasFilledTimes(row)) {
      if ((index + 1) % 25 === 0 || index === records.length - 1) {
        console.log(`Skipped ${index + 1}/${records.length} rows`);
      }
      continue;
    }

    let estimateSeconds;
    let predictSeconds;
    try {
      [estimateSeconds, predictSeconds] = await Promise.all([
        getEstimateSeconds(args, row, rowNumber, departureTime, estimateCache),
        getPredictSeconds(args, row, rowNumber, departureTime, predictCache),
      ]);
    } catch (error) {
      await writeOutput(args, headers, records);
      throw new Error(`Row ${rowNumber} (${row.timestamp}, hour ${row.hour}) failed: ${error.message}`);
    }

    row.estimate_time = formatSeconds(estimateSeconds, args.decimals);
    row.predict_time = formatSeconds(predictSeconds, args.decimals);

    if ((index + 1) % args.saveEvery === 0) {
      await writeOutput(args, headers, records);
    }

    if ((index + 1) % 25 === 0 || index === records.length - 1) {
      console.log(`Filled ${index + 1}/${records.length} rows`);
    }
  }

  await writeOutput(args, headers, records);
  console.log(`Wrote ${args.output}`);
  console.log(`Vietmap calls: ${estimateCache.size}; model calls: ${predictCache.size}; dropped rows: ${droppedRows}`);
}

main().catch((error) => {
  console.error(error instanceof Error ? error.message : error);
  process.exitCode = 1;
});
