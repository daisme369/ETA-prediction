import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def flatten_record(record: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}

    for key, value in record.items():
        column_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            flattened.update(flatten_record(value, prefix=f"{column_name}."))
        elif isinstance(value, list):
            flattened[column_name] = json.dumps(value, ensure_ascii=False)
        else:
            flattened[column_name] = value

    return flattened


def load_json_records(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        if len(data) == 1 and isinstance(next(iter(data.values())), list):
            records = next(iter(data.values()))
        else:
            records = [data]
    else:
        raise ValueError("JSON input must be an object or an array of objects.")

    normalized_records: List[Dict[str, Any]] = []
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every JSON record must be an object.")
        normalized_records.append(flatten_record(record))

    return normalized_records


def write_csv(records: Iterable[Dict[str, Any]], output_path: Path) -> None:
    records = list(records)
    if not records:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames: List[str] = sorted({key for record in records for key in record.keys()})

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert JSON records to CSV.")
    parser.add_argument("input", help="Path to the input JSON file")
    parser.add_argument("output", help="Path to the output CSV file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    records = load_json_records(input_path)
    write_csv(records, output_path)


if __name__ == "__main__":
    main()