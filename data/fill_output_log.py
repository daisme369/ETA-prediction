from __future__ import annotations

import argparse
import csv
import json
import os
import ssl
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo


DEFAULT_INPUT = Path("data/eda_outputs/sample_with_segments.csv")
DEFAULT_OUTPUT = Path("data/output_log.csv")
DEFAULT_ROUTE_URL = "https://maps.vietmap.vn/api/route/v3"
DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"

OUTPUT_COLUMNS = [
    "timestamp",
    "stationId",
    "destination_stationId",
    "lat",
    "lng",
    "destination_lat",
    "destination_lng",
    "estimate_time",
    "distance_meters",
    "vehicle",
    "request_time_iso",
    "vietmap_status",
    "error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crawl Vietmap ETA baseline for each segment row. The datetime column is "
            "interpreted in the configured local timezone and sent to Vietmap as the "
            "route departure time."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--api-key", default=os.getenv("VIETMAP_API_KEY", ""))
    parser.add_argument("--route-url", default=os.getenv("VIETMAP_ROUTE_URL", DEFAULT_ROUTE_URL))
    parser.add_argument("--timezone", default=os.getenv("ETA_DATASET_TIMEZONE", DEFAULT_TIMEZONE))
    parser.add_argument("--vehicle", default=os.getenv("VIETMAP_DEFAULT_VEHICLE", "car"))
    parser.add_argument("--sleep-secs", type=float, default=0.15)
    parser.add_argument("--timeout-secs", type=float, default=20.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true", help="Ignore existing output rows and crawl from scratch.")
    parser.add_argument(
        "--insecure-skip-tls-verify",
        action="store_true",
        help="Disable HTTPS certificate verification. Use only when local Python CA certificates are broken.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate input and print the first request without calling API.")
    return parser.parse_args()


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"')
        os.environ.setdefault(key, value)


def required_value(row: dict[str, str], key: str, row_number: int) -> str:
    value = (row.get(key) or "").strip()
    if not value:
        raise ValueError(f"Row {row_number} is missing {key}.")
    return value


def parse_float(row: dict[str, str], key: str, row_number: int) -> float:
    try:
        return float(required_value(row, key, row_number))
    except ValueError as exc:
        raise ValueError(f"Row {row_number} has invalid {key}: {row.get(key)!r}") from exc


def parse_dataset_datetime(row: dict[str, str], row_number: int, timezone_name: str) -> tuple[str, str]:
    raw = required_value(row, "datetime", row_number)
    timestamp = datetime.strptime(raw, "%Y-%m-%d %H:%M:%S")
    local_time = timestamp.replace(tzinfo=ZoneInfo(timezone_name))
    return timestamp.strftime("%Y-%m-%d %H:%M:%S"), local_time.isoformat()


def row_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("timestamp") or row.get("datetime") or ""),
        str(row.get("stationId") or ""),
        str(row.get("destination_stationId") or ""),
    )


def read_input_rows(path: Path, limit: int | None) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    if limit is not None:
        rows = rows[:limit]
    return rows


def read_completed_rows(path: Path) -> tuple[set[tuple[str, str, str]], list[dict[str, str]]]:
    if not path.exists():
        return set(), []

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
    successful_rows = [row for row in rows if not row.get("error") and row.get("estimate_time")]
    return {row_key(row) for row in successful_rows}, successful_rows


def build_route_url(
    route_url: str,
    api_key: str,
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    vehicle: str,
    request_time_iso: str,
) -> str:
    params: list[tuple[str, str]] = [
        ("apikey", api_key),
        ("point", f"{origin_lat:.6f},{origin_lng:.6f}"),
        ("point", f"{destination_lat:.6f},{destination_lng:.6f}"),
        ("vehicle", vehicle),
        ("points_encoded", "false"),
        ("annotations", "congestion,congestion_distance"),
        ("time", request_time_iso),
    ]
    return f"{route_url}?{urlencode(params)}"


def build_ssl_context(insecure_skip_tls_verify: bool) -> ssl.SSLContext:
    if insecure_skip_tls_verify:
        return ssl._create_unverified_context()

    try:
        import certifi  # type: ignore[import-not-found]

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_json(url: str, timeout_secs: float, ssl_context: ssl.SSLContext) -> dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "eta-baseline-crawler/1.0"})
    with urlopen(request, timeout=timeout_secs, context=ssl_context) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_vietmap_eta(
    url: str,
    timeout_secs: float,
    max_retries: int,
    ssl_context: ssl.SSLContext,
) -> tuple[float | None, float | None, str, str]:
    last_error = ""
    for attempt in range(max_retries + 1):
        try:
            payload = fetch_json(url, timeout_secs, ssl_context)
            status = str(payload.get("code", ""))
            paths = payload.get("paths")
            if status != "OK" or not isinstance(paths, list) or not paths:
                return None, None, status, json.dumps(payload, ensure_ascii=False)[:1000]

            primary_path = paths[0]
            duration_ms = primary_path.get("time")
            distance_meters = primary_path.get("distance")
            if duration_ms is None:
                return None, None, status, "Vietmap response path is missing time."
            return float(duration_ms) / 1000.0, float(distance_meters or 0.0), status, ""
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")[:1000]
            last_error = f"HTTP {exc.code}: {body}"
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = str(exc)

        if attempt < max_retries:
            time.sleep(min(2.0 * (attempt + 1), 8.0))

    return None, None, "", last_error


def make_output_row(
    source_row: dict[str, str],
    row_number: int,
    timezone_name: str,
    route_url: str,
    api_key: str,
    vehicle: str,
    timeout_secs: float,
    max_retries: int,
    ssl_context: ssl.SSLContext,
    dry_run: bool,
) -> dict[str, str]:
    timestamp, request_time_iso = parse_dataset_datetime(source_row, row_number, timezone_name)
    origin_lat = parse_float(source_row, "lat", row_number)
    origin_lng = parse_float(source_row, "lng", row_number)
    destination_lat = parse_float(source_row, "destination_lat", row_number)
    destination_lng = parse_float(source_row, "destination_lng", row_number)

    url = build_route_url(
        route_url,
        api_key,
        origin_lat,
        origin_lng,
        destination_lat,
        destination_lng,
        vehicle,
        request_time_iso,
    )
    if dry_run:
        print(url.replace(api_key, "***"))
        estimate_time = None
        distance_meters = None
        status = "DRY_RUN"
        error = ""
    else:
        estimate_time, distance_meters, status, error = fetch_vietmap_eta(url, timeout_secs, max_retries, ssl_context)

    return {
        "timestamp": timestamp,
        "stationId": required_value(source_row, "stationId", row_number),
        "destination_stationId": required_value(source_row, "destination_stationId", row_number),
        "lat": f"{origin_lat:.10f}",
        "lng": f"{origin_lng:.10f}",
        "destination_lat": f"{destination_lat:.10f}",
        "destination_lng": f"{destination_lng:.10f}",
        "estimate_time": "" if estimate_time is None else f"{estimate_time:.6f}",
        "distance_meters": "" if distance_meters is None else f"{distance_meters:.3f}",
        "vehicle": vehicle,
        "request_time_iso": request_time_iso,
        "vietmap_status": status,
        "error": error,
    }


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    load_env_file(Path(".env"))
    args = parse_args()
    if not args.api_key:
        print("Missing VIETMAP_API_KEY. Set it in .env or pass --api-key.", file=sys.stderr)
        return 2

    source_rows = read_input_rows(args.input, args.limit)
    completed_keys, output_rows = (set(), []) if args.force else read_completed_rows(args.output)
    ssl_context = build_ssl_context(args.insecure_skip_tls_verify)

    total = len(source_rows)
    crawled = 0
    skipped = 0
    failed = 0
    for index, source_row in enumerate(source_rows, start=1):
        timestamp, _ = parse_dataset_datetime(source_row, index, args.timezone)
        key = (timestamp, str(source_row.get("stationId") or ""), str(source_row.get("destination_stationId") or ""))
        if key in completed_keys:
            skipped += 1
            continue

        output_row = make_output_row(
            source_row,
            index,
            args.timezone,
            args.route_url,
            args.api_key,
            args.vehicle,
            args.timeout_secs,
            args.max_retries,
            ssl_context,
            args.dry_run,
        )
        output_rows.append(output_row)
        crawled += 1
        if output_row["error"]:
            failed += 1

        write_rows(args.output, output_rows)
        print(
            f"[{index}/{total}] {output_row['timestamp']} "
            f"{output_row['stationId']}->{output_row['destination_stationId']} "
            f"eta={output_row['estimate_time'] or 'NA'} status={output_row['vietmap_status']}"
            f"{' error=' + output_row['error'][:180] if output_row['error'] else ''}",
            flush=True,
        )

        if args.dry_run:
            break
        if args.sleep_secs > 0:
            time.sleep(args.sleep_secs)

    print(f"Done. crawled={crawled} skipped={skipped} failed={failed} output={args.output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
