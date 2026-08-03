"""Download NOAA daily weather observations.

The script reads station, variable, date-range, and output settings from
``config/data_sources.yml``. It preserves the raw CSV response, performs
structural and coverage checks, and writes a metadata JSON file for each
configured station.

Usage
-----
From the repository root:

    python src/data/download_noaa_weather.py

To replace an existing snapshot whose contents differ:

    python src/data/download_noaa_weather.py --overwrite
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
    """Load and validate the top-level YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a YAML mapping.")

    return config


def calculate_sha256(content: bytes) -> str:
    """Return the SHA-256 checksum for a byte sequence."""
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


def bool_to_api_string(value: bool) -> str:
    """Convert a Python boolean to the lowercase form expected by the API."""
    return "true" if value else "false"


def download_station_csv(
    endpoint: str,
    parameters: dict[str, Any],
    timeout_seconds: int,
) -> requests.Response:
    """Download one station's CSV response."""
    response = requests.get(
        endpoint,
        params=parameters,
        timeout=timeout_seconds,
        headers={
            "User-Agent": (
                "municipal-water-demand-forecasting/1.0 "
                "(public research and portfolio project)"
            )
        },
    )

    response.raise_for_status()

    if not response.content.strip():
        raise RuntimeError("NOAA returned an empty response.")

    beginning = response.content[:500].lower()

    if b"<html" in beginning or b"<!doctype html" in beginning:
        raise RuntimeError("NOAA returned HTML instead of CSV data.")

    return response


def inspect_station_csv(
    content: bytes,
    station_id: str,
    start_date: str,
    end_date: str,
    required_variables: list[str],
    optional_variables: list[str],
    minimum_required_coverage: float,
    require_complete_calendar: bool,
    allow_duplicate_dates: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate schema, dates, station identity, and variable coverage."""
    try:
        dataframe = pd.read_csv(io.BytesIO(content))
    except Exception as error:
        raise ValueError("The NOAA response is not a valid CSV.") from error

    if dataframe.empty:
        raise ValueError(f"NOAA returned no rows for station {station_id}.")

    structural_columns = ["STATION", "DATE"]
    missing_structural_columns = [
        column
        for column in structural_columns
        if column not in dataframe.columns
    ]

    if missing_structural_columns:
        raise ValueError(
            "The NOAA response is missing structural columns: "
            + ", ".join(missing_structural_columns)
        )

    missing_required_columns = [
        variable
        for variable in required_variables
        if variable not in dataframe.columns
    ]

    if missing_required_columns:
        raise ValueError(
            "The NOAA response is missing required variables: "
            + ", ".join(missing_required_columns)
        )

    returned_stations = sorted(
        dataframe["STATION"].dropna().astype(str).unique().tolist()
    )

    if returned_stations != [station_id]:
        raise ValueError(
            f"Expected station {station_id}, but received "
            f"{returned_stations}."
        )

    parsed_dates = pd.to_datetime(dataframe["DATE"], errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())

    if invalid_date_count:
        raise ValueError(
            f"The DATE column contains {invalid_date_count} invalid values."
        )

    normalized_dates = parsed_dates.dt.normalize()
    duplicate_date_count = int(normalized_dates.duplicated().sum())

    if duplicate_date_count and not allow_duplicate_dates:
        raise ValueError(
            f"Station {station_id} contains "
            f"{duplicate_date_count} duplicate dates."
        )

    expected_dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D",
    )

    observed_dates = pd.DatetimeIndex(normalized_dates.unique()).sort_values()
    missing_calendar_dates = expected_dates.difference(observed_dates)
    unexpected_dates = observed_dates.difference(expected_dates)

    if require_complete_calendar and len(missing_calendar_dates) > 0:
        first_missing = [
            date.date().isoformat()
            for date in missing_calendar_dates[:10]
        ]

        raise ValueError(
            f"Station {station_id} is missing "
            f"{len(missing_calendar_dates)} calendar dates. "
            f"First missing dates: {first_missing}"
        )

    all_variables = required_variables + optional_variables
    coverage_by_variable: dict[str, dict[str, Any]] = {}

    for variable in all_variables:
        if variable not in dataframe.columns:
            coverage_by_variable[variable] = {
                "present": False,
                "nonmissing_count": 0,
                "missing_count": int(len(dataframe)),
                "coverage_fraction": 0.0,
                "coverage_percent": 0.0,
            }
            continue

        numeric_values = pd.to_numeric(
            dataframe[variable],
            errors="coerce",
        )

        nonmissing_count = int(numeric_values.notna().sum())
        missing_count = int(numeric_values.isna().sum())
        coverage_fraction = nonmissing_count / len(dataframe)

        coverage_by_variable[variable] = {
            "present": True,
            "nonmissing_count": nonmissing_count,
            "missing_count": missing_count,
            "coverage_fraction": round(coverage_fraction, 6),
            "coverage_percent": round(100 * coverage_fraction, 2),
        }

        if (
            variable in required_variables
            and coverage_fraction < minimum_required_coverage
        ):
            raise ValueError(
                f"Required variable {variable} has "
                f"{coverage_fraction:.2%} coverage, below the configured "
                f"minimum of {minimum_required_coverage:.2%}."
            )

    summary = {
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "columns": dataframe.columns.tolist(),
        "returned_stations": returned_stations,
        "earliest_date": normalized_dates.min().date().isoformat(),
        "latest_date": normalized_dates.max().date().isoformat(),
        "expected_date_count": int(len(expected_dates)),
        "unique_date_count": int(normalized_dates.nunique()),
        "invalid_date_count": invalid_date_count,
        "duplicate_date_count": duplicate_date_count,
        "missing_calendar_date_count": int(len(missing_calendar_dates)),
        "unexpected_date_count": int(len(unexpected_dates)),
        "missing_calendar_dates": [
            date.date().isoformat()
            for date in missing_calendar_dates
        ],
        "unexpected_dates": [
            date.date().isoformat()
            for date in unexpected_dates
        ],
        "missing_values_by_column": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "coverage_by_variable": coverage_by_variable,
    }

    return dataframe, summary


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Download NOAA daily weather observations."
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
    """Run the NOAA download and metadata workflow."""
    arguments = parse_arguments()
    repository_root = get_repository_root()

    config_path = (
        arguments.config
        if arguments.config is not None
        else repository_root / "config" / "data_sources.yml"
    )

    config = load_config(config_path)

    project_config = config["project"]
    source_config = config["sources"]["noaa_weather"]
    access_config = source_config["access"]
    request_config = source_config["request"]
    validation_config = source_config.get("validation", {})
    output_config = source_config["output"]
    date_range = source_config["date_range"]

    endpoint = access_config["endpoint"]
    required_variables = source_config["variables"]["required"]
    optional_variables = source_config["variables"].get("optional", [])
    requested_variables = required_variables + optional_variables

    project_timezone = ZoneInfo(
        project_config.get("timezone", "America/Denver")
    )
    retrieved_at_utc = datetime.now(timezone.utc)
    retrieved_at_local = retrieved_at_utc.astimezone(project_timezone)
    download_date = retrieved_at_local.date().isoformat()

    output_directory = repository_root / output_config["directory"]
    output_directory.mkdir(parents=True, exist_ok=True)

    configured_overwrite = bool(
        project_config.get("downloads", {}).get(
            "overwrite_existing",
            False,
        )
    )
    overwrite_allowed = arguments.overwrite or configured_overwrite

    save_metadata = bool(
        project_config.get("downloads", {}).get(
            "save_response_metadata",
            True,
        )
    )

    for station in source_config["stations"]:
        station_id = station["id"]

        parameters = {
            "dataset": source_config["dataset_id"],
            "stations": station_id,
            "startDate": date_range["start_date"],
            "endDate": date_range["end_date"],
            "dataTypes": ",".join(requested_variables),
            "format": access_config.get("format", "csv"),
            "units": access_config.get("units", "standard"),
            "includeAttributes": bool_to_api_string(
                request_config.get("include_attributes", True)
            ),
            "includeStationName": bool_to_api_string(
                request_config.get("include_station_name", True)
            ),
            "includeStationLocation": bool_to_api_string(
                request_config.get("include_station_location", True)
            ),
        }

        response = download_station_csv(
            endpoint=endpoint,
            parameters=parameters,
            timeout_seconds=request_config.get("timeout_seconds", 120),
        )

        _, validation_summary = inspect_station_csv(
            content=response.content,
            station_id=station_id,
            start_date=date_range["start_date"],
            end_date=date_range["end_date"],
            required_variables=required_variables,
            optional_variables=optional_variables,
            minimum_required_coverage=float(
                validation_config.get(
                    "minimum_required_coverage",
                    0.99,
                )
            ),
            require_complete_calendar=bool(
                validation_config.get(
                    "require_complete_calendar",
                    True,
                )
            ),
            allow_duplicate_dates=bool(
                validation_config.get(
                    "allow_duplicate_dates",
                    False,
                )
            ),
        )

        csv_filename = output_config["filename_template"].format(
            station_id=station_id,
            download_date=download_date,
        )
        metadata_filename = output_config[
            "metadata_filename_template"
        ].format(
            station_id=station_id,
            download_date=download_date,
        )

        csv_path = output_directory / csv_filename
        metadata_path = output_directory / metadata_filename

        downloaded_checksum = calculate_sha256(response.content)
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

        metadata = {
            "config_version": config.get("config_version"),
            "dataset_name": source_config["dataset_name"],
            "dataset_id": source_config["dataset_id"],
            "provider": source_config["provider"],
            "station": station,
            "selected_method": access_config["method"],
            "requested_endpoint": endpoint,
            "request_url": response.url,
            "http_status": response.status_code,
            "content_type": response.headers.get("Content-Type"),
            "retrieved_at_utc": retrieved_at_utc.isoformat(),
            "retrieved_at_local": retrieved_at_local.isoformat(),
            "download_date": download_date,
            "configured_start_date": date_range["start_date"],
            "configured_end_date": date_range["end_date"],
            "requested_variables": requested_variables,
            "required_variables": required_variables,
            "optional_variables": optional_variables,
            "output_file": csv_path.relative_to(
                repository_root
            ).as_posix(),
            "file_action": file_action,
            "file_size_bytes": len(response.content),
            "sha256": downloaded_checksum,
            "query_parameters": parameters,
            **validation_summary,
        }

        if save_metadata:
            write_json_atomically(metadata_path, metadata)

        print(f"NOAA download completed for {station_id}.")
        print(f"Data file: {csv_path}")
        print(f"File action: {file_action}")
        print(f"Rows: {validation_summary['row_count']:,}")
        print(
            "Date range: "
            f"{validation_summary['earliest_date']} through "
            f"{validation_summary['latest_date']}"
        )
        print(f"SHA-256: {downloaded_checksum}")

        for variable, coverage in validation_summary[
            "coverage_by_variable"
        ].items():
            print(
                f"{variable}: "
                f"{coverage['coverage_percent']:.2f}% coverage"
            )

        if save_metadata:
            print(f"Metadata file: {metadata_path}")

        print()


if __name__ == "__main__":
    main()
