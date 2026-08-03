"""Build the analysis-ready daily water-demand and weather dataset.

The script reads the newest validated raw snapshots, applies canonical column
names, joins water demand to NOAA weather by date, creates deterministic
weather and calendar features, and writes a processed CSV with provenance
metadata.

Usage
-----
From the repository root:

    python src/data/build_daily_dataset.py

To replace an existing processed file whose contents differ:

    python src/data/build_daily_dataset.py --overwrite
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


DEFAULT_WEATHER_NAMES = {
    "TMAX": "tmax_f",
    "TMIN": "tmin_f",
    "PRCP": "precipitation_in",
    "SNOW": "snowfall_in",
    "SNWD": "snow_depth_in",
}


def get_repository_root() -> Path:
    """Return the repository root based on this script's location."""
    return Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML mapping."""
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = yaml.safe_load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")

    return content


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object."""
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        content = json.load(file)

    if not isinstance(content, dict):
        raise ValueError(f"Expected a JSON object in {path}.")

    return content


def calculate_sha256_bytes(content: bytes) -> str:
    """Return the SHA-256 checksum of a byte sequence."""
    return hashlib.sha256(content).hexdigest()


def calculate_sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


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


def find_latest_metadata(
    directory: Path,
    *,
    station_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Find the newest matching acquisition metadata file."""
    candidates: list[tuple[datetime, Path, dict[str, Any]]] = []

    for path in directory.glob("*_metadata.json"):
        try:
            metadata = load_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

        if station_id is not None:
            returned_station = metadata.get("station", {}).get("id")
            if returned_station != station_id:
                continue

        timestamp_text = metadata.get("retrieved_at_utc")

        try:
            timestamp = datetime.fromisoformat(timestamp_text)
        except (TypeError, ValueError):
            timestamp = datetime.fromtimestamp(
                path.stat().st_mtime,
                tz=timezone.utc,
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        candidates.append((timestamp, path, metadata))

    if not candidates:
        station_note = f" for station {station_id}" if station_id else ""
        raise FileNotFoundError(
            f"No acquisition metadata found in {directory}{station_note}."
        )

    _, path, metadata = max(candidates, key=lambda item: item[0])
    return path, metadata


def resolve_raw_csv(
    repository_root: Path,
    metadata: dict[str, Any],
) -> Path:
    """Resolve and verify the raw CSV referenced by acquisition metadata."""
    relative_path = metadata.get("output_file")

    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError("Acquisition metadata is missing output_file.")

    path = repository_root / relative_path

    if not path.exists():
        raise FileNotFoundError(f"Raw CSV referenced by metadata is missing: {path}")

    expected_checksum = metadata.get("sha256")
    actual_checksum = calculate_sha256_file(path)

    if actual_checksum != expected_checksum:
        raise ValueError(
            f"Raw CSV checksum mismatch for {path}. "
            "Run validate_raw_data.py before building the processed dataset."
        )

    return path


def require_passing_validation(report: dict[str, Any]) -> None:
    """Stop if the saved raw-data validation report contains errors."""
    error_count = report.get("summary", {}).get("error_count")
    status = report.get("status")

    if error_count != 0 or status not in {"pass", "pass_with_warnings"}:
        raise ValueError(
            "The raw-data validation report is not passing. "
            "Run src/data/validate_raw_data.py and resolve all errors first."
        )


def prepare_water_data(
    dataframe: pd.DataFrame,
    source_config: dict[str, Any],
) -> pd.DataFrame:
    """Select and rename configured water-demand fields."""
    fields = source_config["fields"]
    rename_map = {
        field["source_name"]: field["canonical_name"]
        for field in fields.values()
    }
    required_source_columns = list(rename_map)
    missing_columns = [
        column for column in required_source_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Water-demand data are missing configured columns: "
            + ", ".join(missing_columns)
        )

    result = dataframe[required_source_columns].rename(columns=rename_map).copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()

    for column in result.columns:
        if column != "date":
            result[column] = pd.to_numeric(result[column], errors="raise")

    if result["date"].duplicated().any():
        raise ValueError("Water-demand data contain duplicate dates.")

    return result.sort_values("date").reset_index(drop=True)


def prepare_weather_data(
    dataframe: pd.DataFrame,
    weather_names: dict[str, str],
    station_id: str,
) -> pd.DataFrame:
    """Select and rename the initial weather variables."""
    required_columns = ["STATION", "DATE", *weather_names.keys()]
    missing_columns = [
        column for column in required_columns if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "NOAA weather data are missing selected columns: "
            + ", ".join(missing_columns)
        )

    returned_stations = sorted(
        dataframe["STATION"].dropna().astype(str).unique().tolist()
    )
    if returned_stations != [station_id]:
        raise ValueError(
            f"Expected NOAA station {station_id}, received {returned_stations}."
        )

    result = dataframe[["DATE", *weather_names.keys()]].rename(
        columns={"DATE": "date", **weather_names}
    ).copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()

    for column in weather_names.values():
        result[column] = pd.to_numeric(result[column], errors="raise")

    if result["date"].duplicated().any():
        raise ValueError("NOAA weather data contain duplicate dates.")

    if result[list(weather_names.values())].isna().any().any():
        missing = {
            column: int(count)
            for column, count in result[list(weather_names.values())]
            .isna()
            .sum()
            .items()
            if count
        }
        raise ValueError(
            "Selected modeling weather variables contain missing values: "
            f"{missing}"
        )

    return result.sort_values("date").reset_index(drop=True)


def add_derived_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic weather and calendar features."""
    result = dataframe.copy()

    result["tavg_f"] = (result["tmax_f"] + result["tmin_f"]) / 2.0
    result["temperature_range_f"] = result["tmax_f"] - result["tmin_f"]

    result["year"] = result["date"].dt.year.astype("int16")
    result["month"] = result["date"].dt.month.astype("int8")
    result["day_of_year"] = result["date"].dt.dayofyear.astype("int16")
    result["day_of_week"] = result["date"].dt.dayofweek.astype("int8")
    result["is_weekend"] = (result["day_of_week"] >= 5).astype("int8")

    return result


def serialize_csv(dataframe: pd.DataFrame) -> bytes:
    """Serialize the processed table deterministically as UTF-8 CSV."""
    text = dataframe.to_csv(
        index=False,
        date_format="%Y-%m-%d",
        lineterminator="\n",
        float_format="%.15g",
    )
    return text.encode("utf-8")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Build the processed daily water-demand dataset."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to data_sources.yml.",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=None,
        help="Optional path to raw_data_validation.json.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing processed file if its contents differ.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save the processed daily dataset."""
    arguments = parse_arguments()
    repository_root = get_repository_root()

    config_path = (
        arguments.config
        if arguments.config is not None
        else repository_root / "config" / "data_sources.yml"
    )
    config = load_yaml(config_path)

    processing_config = config.get("processing", {}).get("daily_dataset", {})
    validation_report_path = (
        arguments.validation_report
        if arguments.validation_report is not None
        else repository_root
        / processing_config.get(
            "validation_report",
            "reports/raw_data_validation.json",
        )
    )
    validation_report = load_json(validation_report_path)
    require_passing_validation(validation_report)

    water_config = config["sources"]["water_demand"]
    weather_config = config["sources"]["noaa_weather"]

    water_metadata_path, water_metadata = find_latest_metadata(
        repository_root / water_config["output"]["directory"]
    )

    primary_stations = [
        station
        for station in weather_config["stations"]
        if station.get("role") == "primary"
    ]
    station = primary_stations[0] if primary_stations else weather_config["stations"][0]
    station_id = station["id"]

    weather_metadata_path, weather_metadata = find_latest_metadata(
        repository_root / weather_config["output"]["directory"],
        station_id=station_id,
    )

    water_csv_path = resolve_raw_csv(repository_root, water_metadata)
    weather_csv_path = resolve_raw_csv(repository_root, weather_metadata)

    water_raw = pd.read_csv(water_csv_path)
    weather_raw = pd.read_csv(weather_csv_path)

    weather_names = processing_config.get(
        "weather_variables",
        DEFAULT_WEATHER_NAMES,
    )
    if not isinstance(weather_names, dict) or not weather_names:
        raise ValueError("processing.daily_dataset.weather_variables must be a mapping.")

    water = prepare_water_data(water_raw, water_config)
    weather = prepare_weather_data(weather_raw, weather_names, station_id)

    join_method = processing_config.get("join_method", "inner")
    if join_method != "inner":
        raise ValueError("The initial daily dataset currently requires join_method: inner.")

    combined = water.merge(
        weather,
        on="date",
        how=join_method,
        validate="one_to_one",
        indicator=True,
    )

    unmatched_count = int((combined["_merge"] != "both").sum())
    combined = combined.drop(columns="_merge")

    if unmatched_count:
        raise ValueError(f"The join produced {unmatched_count} unmatched rows.")

    if len(combined) != len(water) or len(combined) != len(weather):
        raise ValueError(
            "The joined row count differs from one or both validated inputs."
        )

    combined = add_derived_features(combined)

    ordered_columns = [
        "date",
        "water_demand_mgd",
        "projected_water_demand_mgd",
        "actual_projected_demand_ratio",
        "plant_demand_mgd",
        "tmax_f",
        "tmin_f",
        "tavg_f",
        "temperature_range_f",
        "precipitation_in",
        "snowfall_in",
        "snow_depth_in",
        "year",
        "month",
        "day_of_year",
        "day_of_week",
        "is_weekend",
    ]

    missing_output_columns = [
        column for column in ordered_columns if column not in combined.columns
    ]
    if missing_output_columns:
        raise ValueError(
            "Processed dataset is missing expected columns: "
            + ", ".join(missing_output_columns)
        )

    combined = combined[ordered_columns].sort_values("date").reset_index(drop=True)

    if combined.isna().any().any():
        missing = {
            column: int(count)
            for column, count in combined.isna().sum().items()
            if count
        }
        raise ValueError(f"Processed dataset contains missing values: {missing}")

    if combined["date"].duplicated().any():
        raise ValueError("Processed dataset contains duplicate dates.")

    expected_dates = pd.date_range(
        start=combined["date"].min(),
        end=combined["date"].max(),
        freq="D",
    )
    observed_dates = pd.DatetimeIndex(combined["date"])
    missing_dates = expected_dates.difference(observed_dates)
    if len(missing_dates):
        raise ValueError(
            f"Processed dataset is missing {len(missing_dates)} calendar dates."
        )

    output_config = processing_config.get("output", {})
    output_directory = repository_root / output_config.get(
        "directory",
        "data/processed",
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    csv_path = output_directory / output_config.get(
        "filename",
        "fort_collins_daily_water_weather.csv",
    )
    metadata_path = output_directory / output_config.get(
        "metadata_filename",
        "fort_collins_daily_water_weather_metadata.json",
    )

    csv_content = serialize_csv(combined)
    output_checksum = calculate_sha256_bytes(csv_content)

    overwrite_allowed = arguments.overwrite or bool(
        config.get("project", {})
        .get("downloads", {})
        .get("overwrite_existing", False)
    )
    file_action = "created"

    if csv_path.exists():
        existing_checksum = calculate_sha256_file(csv_path)
        if existing_checksum == output_checksum:
            file_action = "confirmed_existing_file"
        elif overwrite_allowed:
            write_bytes_atomically(csv_path, csv_content)
            file_action = "overwritten"
        else:
            raise FileExistsError(
                f"A different processed file already exists at {csv_path}. "
                "Use --overwrite only after reviewing the difference."
            )
    else:
        write_bytes_atomically(csv_path, csv_content)

    generated_at_utc = datetime.now(timezone.utc)
    target_column = processing_config.get("target_column", "water_demand_mgd")
    default_feature_columns = [
        "tmax_f",
        "tmin_f",
        "tavg_f",
        "temperature_range_f",
        "precipitation_in",
        "snowfall_in",
        "snow_depth_in",
        "year",
        "month",
        "day_of_year",
        "day_of_week",
        "is_weekend",
    ]
    reference_columns = [
        "projected_water_demand_mgd",
        "actual_projected_demand_ratio",
        "plant_demand_mgd",
    ]

    metadata = {
        "config_version": config.get("config_version"),
        "generated_at_utc": generated_at_utc.isoformat(),
        "validation_report": validation_report_path.relative_to(
            repository_root
        ).as_posix(),
        "validation_status": validation_report.get("status"),
        "inputs": {
            "water_csv": water_csv_path.relative_to(repository_root).as_posix(),
            "water_metadata": water_metadata_path.relative_to(
                repository_root
            ).as_posix(),
            "water_sha256": water_metadata.get("sha256"),
            "weather_csv": weather_csv_path.relative_to(repository_root).as_posix(),
            "weather_metadata": weather_metadata_path.relative_to(
                repository_root
            ).as_posix(),
            "weather_sha256": weather_metadata.get("sha256"),
            "weather_station_id": station_id,
        },
        "output_file": csv_path.relative_to(repository_root).as_posix(),
        "file_action": file_action,
        "file_size_bytes": len(csv_content),
        "sha256": output_checksum,
        "row_count": int(len(combined)),
        "column_count": int(len(combined.columns)),
        "columns": combined.columns.tolist(),
        "earliest_date": combined["date"].min().date().isoformat(),
        "latest_date": combined["date"].max().date().isoformat(),
        "duplicate_date_count": int(combined["date"].duplicated().sum()),
        "missing_calendar_date_count": int(len(missing_dates)),
        "missing_values_by_column": {
            column: int(count)
            for column, count in combined.isna().sum().items()
        },
        "join": {
            "method": join_method,
            "validate": "one_to_one",
            "water_input_rows": int(len(water)),
            "weather_input_rows": int(len(weather)),
            "output_rows": int(len(combined)),
            "unmatched_rows": unmatched_count,
        },
        "target_column": target_column,
        "default_feature_columns": default_feature_columns,
        "reference_columns_not_default_predictors": reference_columns,
        "excluded_raw_variables": {
            "AWND": (
                "Preserved in the NOAA raw snapshot but excluded from the "
                "initial processed dataset because coverage is 83.26%."
            )
        },
        "derived_columns": {
            "tavg_f": "Arithmetic mean of daily TMAX and TMIN.",
            "temperature_range_f": "Daily TMAX minus TMIN.",
            "year": "Calendar year.",
            "month": "Calendar month, 1 through 12.",
            "day_of_year": "Calendar day of year, 1 through 366.",
            "day_of_week": "Monday=0 through Sunday=6.",
            "is_weekend": "1 for Saturday or Sunday, otherwise 0.",
        },
        "units": {
            "water_demand_mgd": "million gallons per day",
            "projected_water_demand_mgd": "million gallons per day",
            "plant_demand_mgd": "million gallons per day",
            "tmax_f": "degrees Fahrenheit",
            "tmin_f": "degrees Fahrenheit",
            "tavg_f": "degrees Fahrenheit",
            "temperature_range_f": "degrees Fahrenheit",
            "precipitation_in": "inches",
            "snowfall_in": "inches",
            "snow_depth_in": "inches",
        },
    }

    write_json_atomically(metadata_path, metadata)

    print("Daily dataset build completed successfully.")
    print(f"Data file: {csv_path}")
    print(f"File action: {file_action}")
    print(f"Rows: {len(combined):,}")
    print(f"Columns: {len(combined.columns)}")
    print(
        "Date range: "
        f"{metadata['earliest_date']} through {metadata['latest_date']}"
    )
    print(f"Missing values: {int(combined.isna().sum().sum())}")
    print(f"SHA-256: {output_checksum}")
    print(f"Metadata file: {metadata_path}")


if __name__ == "__main__":
    main()
