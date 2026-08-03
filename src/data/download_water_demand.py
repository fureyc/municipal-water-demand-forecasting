"""Download the City of Fort Collins daily water-demand dataset.

The script reads source information from ``config/data_sources.yml``,
downloads the raw CSV without modifying its contents, performs basic
structural checks, and writes an accompanying metadata JSON file.

Usage
-----
From the repository root:

    python src/data/download_water_demand.py

To replace an existing snapshot whose contents differ:

    python src/data/download_water_demand.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yaml


def get_repository_root() -> Path:
    """Return the repository root based on this script's location."""
    return Path(__file__).resolve().parents[2]


def load_config(config_path: Path) -> dict[str, Any]:
    """Load the YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


def download_csv(
    endpoints: list[str],
    query_parameters: dict[str, Any],
    timeout_seconds: int,
) -> requests.Response:
    """Try each endpoint in order and return the first successful response."""
    errors: list[str] = []

    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint,
                params=query_parameters,
                timeout=timeout_seconds,
                headers={
                    "User-Agent": (
                        "fort-collins-water-demand-project/1.0 "
                        "(public research and portfolio project)"
                    )
                },
            )
            response.raise_for_status()

            if not response.content.strip():
                raise RuntimeError("The server returned an empty response.")

            beginning = response.content[:500].lower()
            if b"<html" in beginning or b"<!doctype html" in beginning:
                raise RuntimeError(
                    "The endpoint returned HTML instead of CSV data."
                )

            return response

        except (requests.RequestException, RuntimeError) as error:
            errors.append(f"{endpoint}: {error}")

    error_message = "\n".join(errors)
    raise RuntimeError(
        "None of the configured water-demand endpoints succeeded:\n"
        f"{error_message}"
    )


def inspect_csv(
    content: bytes,
    expected_columns: list[str],
    date_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Perform basic structural checks without changing the raw response."""
    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except Exception as error:
        raise ValueError("The downloaded response is not a valid CSV.") from error

    if dataframe.empty:
        raise ValueError("The downloaded CSV contains no data rows.")

    missing_columns = [
        column for column in expected_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "The downloaded CSV is missing expected columns: "
            + ", ".join(missing_columns)
        )

    parsed_dates = pd.to_datetime(dataframe[date_column], errors="coerce")

    invalid_date_count = int(parsed_dates.isna().sum())
    if invalid_date_count:
        raise ValueError(
            f"The date column contains {invalid_date_count} invalid values."
        )

    return dataframe, parsed_dates


def calculate_sha256(content: bytes) -> str:
    """Calculate the SHA-256 checksum of a byte sequence."""
    return hashlib.sha256(content).hexdigest()


def write_bytes_atomically(path: Path, content: bytes) -> None:
    """Write bytes through a temporary file to avoid partial output."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_bytes(content)
    temporary_path.replace(path)


def write_json_atomically(path: Path, content: dict[str, Any]) -> None:
    """Write formatted JSON through a temporary file."""
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download Fort Collins daily water-demand data."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to data_sources.yml.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing snapshot if its contents differ.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the download and metadata workflow."""
    arguments = parse_arguments()
    repository_root = get_repository_root()

    config_path = (
        arguments.config
        if arguments.config is not None
        else repository_root / "config" / "data_sources.yml"
    )

    config = load_config(config_path)

    project_config = config["project"]
    source_config = config["sources"]["water_demand"]
    access_config = source_config["access"]["socrata_legacy"]
    request_config = source_config["request"]
    output_config = source_config["output"]

    endpoints = [
        endpoint
        for endpoint in [
            access_config.get("primary_endpoint"),
            access_config.get("alternate_endpoint"),
        ]
        if endpoint
    ]

    if not endpoints:
        raise ValueError("No Socrata endpoints are configured.")

    query_parameters = {
        "$limit": request_config.get("row_limit", 50000),
        "$order": request_config.get("order_by", "date"),
    }

    response = download_csv(
        endpoints=endpoints,
        query_parameters=query_parameters,
        timeout_seconds=request_config.get("timeout_seconds", 60),
    )

    expected_columns = [
        field["source_name"] for field in source_config["fields"].values()
    ]
    date_column = source_config["fields"]["date"]["source_name"]

    dataframe, parsed_dates = inspect_csv(
        content=response.content,
        expected_columns=expected_columns,
        date_column=date_column,
    )

    project_timezone = ZoneInfo(project_config.get("timezone", "America/Denver"))
    retrieved_at_utc = datetime.now(timezone.utc)
    retrieved_at_local = retrieved_at_utc.astimezone(project_timezone)
    download_date = retrieved_at_local.date().isoformat()

    output_directory = repository_root / output_config["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_filename = output_config["filename_template"].format(
        download_date=download_date
    )
    metadata_filename = output_config["metadata_filename_template"].format(
        download_date=download_date
    )

    csv_path = output_directory / csv_filename
    metadata_path = output_directory / metadata_filename

    downloaded_checksum = calculate_sha256(response.content)
    configured_overwrite = bool(
        project_config.get("downloads", {}).get("overwrite_existing", False)
    )
    overwrite_allowed = arguments.overwrite or configured_overwrite

    file_action = "created"

    if csv_path.exists():
        existing_content = csv_path.read_bytes()
        existing_checksum = calculate_sha256(existing_content)

        if existing_checksum == downloaded_checksum:
            file_action = "confirmed_existing_file"
        elif overwrite_allowed:
            write_bytes_atomically(csv_path, response.content)
            file_action = "overwritten"
        else:
            raise FileExistsError(
                f"A different file already exists at {csv_path}. "
                "Use --overwrite only after reviewing the difference."
            )
    else:
        write_bytes_atomically(csv_path, response.content)

    normalized_dates = parsed_dates.dt.normalize()
    duplicate_date_count = int(normalized_dates.duplicated().sum())

    metadata = {
        "config_version": config.get("config_version"),
        "dataset_name": source_config["dataset_name"],
        "provider": source_config["provider"],
        "selected_method": source_config["access"]["selected_method"],
        "requested_endpoint": endpoints[0],
        "successful_endpoint": response.url.split("?")[0],
        "request_url": response.url,
        "http_status": response.status_code,
        "content_type": response.headers.get("Content-Type"),
        "retrieved_at_utc": retrieved_at_utc.isoformat(),
        "retrieved_at_local": retrieved_at_local.isoformat(),
        "download_date": download_date,
        "output_file": csv_path.relative_to(repository_root).as_posix(),
        "file_action": file_action,
        "file_size_bytes": len(response.content),
        "sha256": downloaded_checksum,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": dataframe.columns.tolist(),
        "earliest_date": normalized_dates.min().date().isoformat(),
        "latest_date": normalized_dates.max().date().isoformat(),
        "duplicate_date_count": duplicate_date_count,
        "missing_values_by_column": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "query_parameters": query_parameters,
    }

    save_metadata = bool(
        project_config.get("downloads", {}).get(
            "save_response_metadata",
            True,
        )
    )

    if save_metadata:
        write_json_atomically(metadata_path, metadata)

    print("Water-demand download completed successfully.")
    print(f"Data file: {csv_path}")
    print(f"File action: {file_action}")
    print(f"Rows: {len(dataframe):,}")
    print(
        "Date range: "
        f"{metadata['earliest_date']} through {metadata['latest_date']}"
    )
    print(f"SHA-256: {downloaded_checksum}")

    if save_metadata:
        print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()
