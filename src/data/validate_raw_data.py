
"""Validate raw water-demand and NOAA weather snapshots.

The script locates the newest metadata file for each configured raw source,
verifies file integrity and schema, checks dataset-specific constraints, and
confirms that the two daily date indexes align exactly.

Usage
-----
From the repository root:

    python src/data/validate_raw_data.py

Optionally choose a different report location:

    python src/data/validate_raw_data.py \
        --output reports/raw_data_validation.json
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


Check = dict[str, Any]


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


def calculate_sha256(path: Path) -> str:
    """Calculate a file's SHA-256 checksum."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def add_check(
    checks: list[Check],
    name: str,
    status: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> None:
    """Append one validation result."""
    if status not in {"pass", "warning", "error"}:
        raise ValueError(f"Unsupported check status: {status}")

    result: Check = {
        "name": name,
        "status": status,
        "message": message,
    }

    if details:
        result["details"] = details

    checks.append(result)


def find_latest_metadata(
    directory: Path,
    *,
    station_id: str | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Find the newest matching metadata file in a raw-data directory."""
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
            f"No metadata files were found in {directory}{station_note}."
        )

    _, path, metadata = max(candidates, key=lambda item: item[0])
    return path, metadata


def resolve_output_file(
    repository_root: Path,
    metadata: dict[str, Any],
) -> Path:
    """Resolve a metadata output_file entry against the repository root."""
    output_file = metadata.get("output_file")

    if not isinstance(output_file, str) or not output_file:
        raise ValueError("Metadata is missing a valid output_file path.")

    path = repository_root / output_file

    if not path.exists():
        raise FileNotFoundError(
            f"Metadata references a missing data file: {path}"
        )

    return path


def validate_metadata_integrity(
    *,
    dataset_label: str,
    csv_path: Path,
    metadata: dict[str, Any],
    dataframe: pd.DataFrame,
    checks: list[Check],
) -> None:
    """Compare the file with its saved acquisition metadata."""
    actual_checksum = calculate_sha256(csv_path)
    expected_checksum = metadata.get("sha256")

    if actual_checksum == expected_checksum:
        add_check(
            checks,
            f"{dataset_label}.checksum",
            "pass",
            "The CSV checksum matches its acquisition metadata.",
        )
    else:
        add_check(
            checks,
            f"{dataset_label}.checksum",
            "error",
            "The CSV checksum does not match its acquisition metadata.",
            {
                "expected_sha256": expected_checksum,
                "actual_sha256": actual_checksum,
            },
        )

    expected_rows = metadata.get("row_count")
    actual_rows = int(len(dataframe))

    add_check(
        checks,
        f"{dataset_label}.row_count",
        "pass" if expected_rows == actual_rows else "error",
        (
            "The CSV row count matches its metadata."
            if expected_rows == actual_rows
            else "The CSV row count does not match its metadata."
        ),
        {
            "expected_row_count": expected_rows,
            "actual_row_count": actual_rows,
        },
    )

    expected_columns = metadata.get("columns")
    actual_columns = dataframe.columns.tolist()

    add_check(
        checks,
        f"{dataset_label}.columns",
        "pass" if expected_columns == actual_columns else "error",
        (
            "The CSV columns match their recorded order in metadata."
            if expected_columns == actual_columns
            else "The CSV columns differ from the metadata."
        ),
        {
            "expected_columns": expected_columns,
            "actual_columns": actual_columns,
        },
    )


def parse_daily_dates(
    dataframe: pd.DataFrame,
    date_column: str,
    dataset_label: str,
    checks: list[Check],
) -> pd.Series:
    """Parse a daily date column and validate uniqueness."""
    if date_column not in dataframe.columns:
        add_check(
            checks,
            f"{dataset_label}.date_column",
            "error",
            f"Required date column {date_column!r} is missing.",
        )
        return pd.Series(dtype="datetime64[ns]")

    parsed = pd.to_datetime(dataframe[date_column], errors="coerce")
    invalid_count = int(parsed.isna().sum())

    add_check(
        checks,
        f"{dataset_label}.valid_dates",
        "pass" if invalid_count == 0 else "error",
        (
            "All date values parsed successfully."
            if invalid_count == 0
            else f"{invalid_count} date values could not be parsed."
        ),
        {"invalid_date_count": invalid_count},
    )

    valid_dates = parsed.dropna().dt.normalize()
    duplicate_count = int(valid_dates.duplicated().sum())

    add_check(
        checks,
        f"{dataset_label}.duplicate_dates",
        "pass" if duplicate_count == 0 else "error",
        (
            "The daily date index contains no duplicates."
            if duplicate_count == 0
            else f"The daily date index contains {duplicate_count} duplicates."
        ),
        {"duplicate_date_count": duplicate_count},
    )

    if valid_dates.empty:
        return valid_dates

    expected = pd.date_range(
        start=valid_dates.min(),
        end=valid_dates.max(),
        freq="D",
    )
    observed = pd.DatetimeIndex(valid_dates.unique()).sort_values()
    missing_dates = expected.difference(observed)

    add_check(
        checks,
        f"{dataset_label}.complete_calendar",
        "pass" if len(missing_dates) == 0 else "error",
        (
            "The dataset contains a complete daily calendar."
            if len(missing_dates) == 0
            else f"The dataset is missing {len(missing_dates)} calendar dates."
        ),
        {
            "earliest_date": valid_dates.min().date().isoformat(),
            "latest_date": valid_dates.max().date().isoformat(),
            "unique_date_count": int(valid_dates.nunique()),
            "missing_calendar_date_count": int(len(missing_dates)),
            "first_missing_dates": [
                date.date().isoformat() for date in missing_dates[:10]
            ],
        },
    )

    return valid_dates


def numeric_series(
    dataframe: pd.DataFrame,
    column: str,
    dataset_label: str,
    checks: list[Check],
    *,
    required: bool,
) -> pd.Series | None:
    """Return a numeric column while reporting invalid values."""
    if column not in dataframe.columns:
        add_check(
            checks,
            f"{dataset_label}.{column}.present",
            "error" if required else "warning",
            (
                f"Required numeric column {column!r} is missing."
                if required
                else f"Optional numeric column {column!r} is missing."
            ),
        )
        return None

    converted = pd.to_numeric(dataframe[column], errors="coerce")
    original_nonmissing = dataframe[column].notna()
    invalid_numeric_count = int((original_nonmissing & converted.isna()).sum())

    add_check(
        checks,
        f"{dataset_label}.{column}.numeric",
        "pass" if invalid_numeric_count == 0 else "error",
        (
            f"Column {column!r} contains only numeric or missing values."
            if invalid_numeric_count == 0
            else (
                f"Column {column!r} contains "
                f"{invalid_numeric_count} non-numeric values."
            )
        ),
        {
            "invalid_numeric_count": invalid_numeric_count,
            "missing_count": int(converted.isna().sum()),
        },
    )

    return converted


def validate_water_demand(
    dataframe: pd.DataFrame,
    source_config: dict[str, Any],
    metadata: dict[str, Any],
    csv_path: Path,
    checks: list[Check],
) -> dict[str, Any]:
    """Validate the Fort Collins water-demand snapshot."""
    label = "water_demand"
    validate_metadata_integrity(
        dataset_label=label,
        csv_path=csv_path,
        metadata=metadata,
        dataframe=dataframe,
        checks=checks,
    )

    configured_fields = source_config["fields"]
    expected_columns = [
        field["source_name"] for field in configured_fields.values()
    ]
    missing_columns = [
        column for column in expected_columns if column not in dataframe.columns
    ]

    add_check(
        checks,
        f"{label}.schema",
        "pass" if not missing_columns else "error",
        (
            "All configured water-demand columns are present."
            if not missing_columns
            else "Configured water-demand columns are missing."
        ),
        {"missing_columns": missing_columns},
    )

    date_column = configured_fields["date"]["source_name"]
    dates = parse_daily_dates(dataframe, date_column, label, checks)

    numeric_columns: dict[str, pd.Series | None] = {}

    for field_name, field_config in configured_fields.items():
        if field_name == "date":
            continue

        source_name = field_config["source_name"]
        numeric_columns[field_name] = numeric_series(
            dataframe,
            source_name,
            label,
            checks,
            required=True,
        )

    for field_name in ("actual_demand", "projected_demand", "plant_demand"):
        values = numeric_columns.get(field_name)

        if values is None:
            continue

        nonpositive_count = int((values <= 0).fillna(False).sum())

        add_check(
            checks,
            f"{label}.{field_name}.positive",
            "pass" if nonpositive_count == 0 else "error",
            (
                f"{field_name} contains only positive observed values."
                if nonpositive_count == 0
                else f"{field_name} contains {nonpositive_count} nonpositive values."
            ),
            {"nonpositive_count": nonpositive_count},
        )

    actual = numeric_columns.get("actual_demand")
    projected = numeric_columns.get("projected_demand")
    recorded_ratio = numeric_columns.get("actual_projected_ratio")

    ratio_mismatch_count = None
    maximum_ratio_error = None

    if actual is not None and projected is not None and recorded_ratio is not None:
        comparable = actual.notna() & projected.notna() & recorded_ratio.notna()
        nonzero_projected = projected != 0
        comparable &= nonzero_projected

        calculated_ratio = actual[comparable] / projected[comparable]
        absolute_error = (
            recorded_ratio[comparable] - calculated_ratio
        ).abs()

        tolerance = 1e-6
        ratio_mismatch_count = int((absolute_error > tolerance).sum())
        maximum_ratio_error = (
            float(absolute_error.max()) if not absolute_error.empty else None
        )

        add_check(
            checks,
            f"{label}.actual_projected_ratio",
            "pass" if ratio_mismatch_count == 0 else "error",
            (
                "The recorded actual/projected ratio is internally consistent."
                if ratio_mismatch_count == 0
                else (
                    f"{ratio_mismatch_count} rows have an inconsistent "
                    "actual/projected ratio."
                )
            ),
            {
                "tolerance": tolerance,
                "ratio_mismatch_count": ratio_mismatch_count,
                "maximum_absolute_error": maximum_ratio_error,
            },
        )

    return {
        "csv_path": csv_path.as_posix(),
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "earliest_date": (
            dates.min().date().isoformat() if not dates.empty else None
        ),
        "latest_date": (
            dates.max().date().isoformat() if not dates.empty else None
        ),
        "unique_date_count": int(dates.nunique()) if not dates.empty else 0,
        "missing_values_by_column": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
        },
        "ratio_mismatch_count": ratio_mismatch_count,
        "maximum_ratio_error": maximum_ratio_error,
    }


def validate_noaa_weather(
    dataframe: pd.DataFrame,
    source_config: dict[str, Any],
    metadata: dict[str, Any],
    csv_path: Path,
    station_id: str,
    checks: list[Check],
) -> dict[str, Any]:
    """Validate one NOAA daily weather snapshot."""
    label = "noaa_weather"
    validate_metadata_integrity(
        dataset_label=label,
        csv_path=csv_path,
        metadata=metadata,
        dataframe=dataframe,
        checks=checks,
    )

    required_structural = ["STATION", "DATE"]
    missing_structural = [
        column for column in required_structural
        if column not in dataframe.columns
    ]

    add_check(
        checks,
        f"{label}.schema",
        "pass" if not missing_structural else "error",
        (
            "Required NOAA structural columns are present."
            if not missing_structural
            else "Required NOAA structural columns are missing."
        ),
        {"missing_columns": missing_structural},
    )

    if "STATION" in dataframe.columns:
        returned_stations = sorted(
            dataframe["STATION"].dropna().astype(str).unique().tolist()
        )
    else:
        returned_stations = []

    add_check(
        checks,
        f"{label}.station_identity",
        "pass" if returned_stations == [station_id] else "error",
        (
            f"All rows belong to configured station {station_id}."
            if returned_stations == [station_id]
            else "Returned station identifiers do not match the configuration."
        ),
        {
            "configured_station": station_id,
            "returned_stations": returned_stations,
        },
    )

    dates = parse_daily_dates(dataframe, "DATE", label, checks)

    configured_start = pd.Timestamp(
        source_config["date_range"]["start_date"]
    )
    configured_end = pd.Timestamp(
        source_config["date_range"]["end_date"]
    )

    observed_start = dates.min() if not dates.empty else None
    observed_end = dates.max() if not dates.empty else None
    matches_configured_range = (
        observed_start == configured_start and observed_end == configured_end
    )

    add_check(
        checks,
        f"{label}.configured_date_range",
        "pass" if matches_configured_range else "error",
        (
            "The NOAA date range matches the configured acquisition period."
            if matches_configured_range
            else "The NOAA date range differs from the configured period."
        ),
        {
            "configured_start_date": configured_start.date().isoformat(),
            "configured_end_date": configured_end.date().isoformat(),
            "observed_start_date": (
                observed_start.date().isoformat()
                if observed_start is not None
                else None
            ),
            "observed_end_date": (
                observed_end.date().isoformat()
                if observed_end is not None
                else None
            ),
        },
    )

    required_variables = source_config["variables"]["required"]
    optional_variables = source_config["variables"].get("optional", [])
    minimum_coverage = float(
        source_config.get("validation", {}).get(
            "minimum_required_coverage",
            0.99,
        )
    )

    coverage: dict[str, dict[str, Any]] = {}
    numeric_values: dict[str, pd.Series | None] = {}

    for variable in required_variables + optional_variables:
        values = numeric_series(
            dataframe,
            variable,
            label,
            checks,
            required=variable in required_variables,
        )
        numeric_values[variable] = values

        if values is None:
            coverage[variable] = {
                "present": False,
                "nonmissing_count": 0,
                "missing_count": int(len(dataframe)),
                "coverage_fraction": 0.0,
                "coverage_percent": 0.0,
            }
            continue

        nonmissing_count = int(values.notna().sum())
        missing_count = int(values.isna().sum())
        coverage_fraction = (
            nonmissing_count / len(dataframe) if len(dataframe) else 0.0
        )

        coverage[variable] = {
            "present": True,
            "nonmissing_count": nonmissing_count,
            "missing_count": missing_count,
            "coverage_fraction": round(coverage_fraction, 6),
            "coverage_percent": round(100 * coverage_fraction, 2),
        }

        if variable in required_variables:
            status = (
                "pass"
                if coverage_fraction >= minimum_coverage
                else "error"
            )
            message = (
                f"{variable} meets the required coverage threshold."
                if status == "pass"
                else f"{variable} falls below the required coverage threshold."
            )
        else:
            status = "pass" if coverage_fraction == 1.0 else "warning"
            message = (
                f"Optional variable {variable} is complete."
                if status == "pass"
                else (
                    f"Optional variable {variable} has "
                    f"{coverage[variable]['coverage_percent']:.2f}% coverage."
                )
            )

        add_check(
            checks,
            f"{label}.{variable}.coverage",
            status,
            message,
            {
                **coverage[variable],
                "required_minimum_coverage": (
                    minimum_coverage
                    if variable in required_variables
                    else None
                ),
            },
        )

    tmax = numeric_values.get("TMAX")
    tmin = numeric_values.get("TMIN")

    if tmax is not None and tmin is not None:
        reversed_temperature_count = int(
            ((tmax < tmin) & tmax.notna() & tmin.notna()).sum()
        )

        add_check(
            checks,
            f"{label}.temperature_order",
            "pass" if reversed_temperature_count == 0 else "error",
            (
                "TMAX is greater than or equal to TMIN on every comparable day."
                if reversed_temperature_count == 0
                else (
                    f"TMAX is below TMIN on "
                    f"{reversed_temperature_count} days."
                )
            ),
            {"reversed_temperature_count": reversed_temperature_count},
        )

        extreme_temperature_count = int(
            (
                ((tmax < -80) | (tmax > 130))
                | ((tmin < -80) | (tmin > 130))
            )
            .fillna(False)
            .sum()
        )

        add_check(
            checks,
            f"{label}.temperature_plausibility",
            "pass" if extreme_temperature_count == 0 else "warning",
            (
                "Temperature values fall within broad plausibility bounds."
                if extreme_temperature_count == 0
                else (
                    f"{extreme_temperature_count} days contain temperatures "
                    "outside broad plausibility bounds."
                )
            ),
            {
                "lower_bound_f": -80,
                "upper_bound_f": 130,
                "flagged_day_count": extreme_temperature_count,
            },
        )

    for variable in ("PRCP", "SNOW", "SNWD", "AWND"):
        values = numeric_values.get(variable)

        if values is None:
            continue

        negative_count = int((values < 0).fillna(False).sum())

        add_check(
            checks,
            f"{label}.{variable}.nonnegative",
            "pass" if negative_count == 0 else "error",
            (
                f"{variable} contains no negative observed values."
                if negative_count == 0
                else f"{variable} contains {negative_count} negative values."
            ),
            {"negative_count": negative_count},
        )

    return {
        "csv_path": csv_path.as_posix(),
        "station_id": station_id,
        "row_count": int(len(dataframe)),
        "column_count": int(len(dataframe.columns)),
        "earliest_date": (
            dates.min().date().isoformat() if not dates.empty else None
        ),
        "latest_date": (
            dates.max().date().isoformat() if not dates.empty else None
        ),
        "unique_date_count": int(dates.nunique()) if not dates.empty else 0,
        "coverage_by_variable": coverage,
        "missing_values_by_column": {
            column: int(count)
            for column, count in dataframe.isna().sum().items()
        },
    }


def validate_cross_source_dates(
    water_dates: pd.Series,
    weather_dates: pd.Series,
    checks: list[Check],
) -> dict[str, Any]:
    """Confirm that the water and weather daily indexes align exactly."""
    water_index = pd.DatetimeIndex(water_dates.unique()).sort_values()
    weather_index = pd.DatetimeIndex(weather_dates.unique()).sort_values()

    water_only = water_index.difference(weather_index)
    weather_only = weather_index.difference(water_index)
    exact_match = len(water_only) == 0 and len(weather_only) == 0

    add_check(
        checks,
        "cross_source.date_alignment",
        "pass" if exact_match else "error",
        (
            "Water-demand and NOAA weather dates align exactly."
            if exact_match
            else "The two raw sources do not have identical daily date indexes."
        ),
        {
            "water_unique_dates": int(len(water_index)),
            "weather_unique_dates": int(len(weather_index)),
            "water_only_date_count": int(len(water_only)),
            "weather_only_date_count": int(len(weather_only)),
            "first_water_only_dates": [
                date.date().isoformat() for date in water_only[:10]
            ],
            "first_weather_only_dates": [
                date.date().isoformat() for date in weather_only[:10]
            ],
        },
    )

    return {
        "exact_date_match": exact_match,
        "water_unique_dates": int(len(water_index)),
        "weather_unique_dates": int(len(weather_index)),
        "water_only_date_count": int(len(water_only)),
        "weather_only_date_count": int(len(weather_only)),
    }


def write_json_atomically(path: Path, content: dict[str, Any]) -> None:
    """Write formatted JSON through a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(content, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Validate raw water-demand and NOAA weather snapshots."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional path to data_sources.yml.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the JSON validation report.",
    )
    return parser.parse_args()


def main() -> None:
    """Run all raw-data validation checks."""
    arguments = parse_arguments()
    repository_root = get_repository_root()

    config_path = (
        arguments.config
        if arguments.config is not None
        else repository_root / "config" / "data_sources.yml"
    )
    output_path = (
        arguments.output
        if arguments.output is not None
        else repository_root / "reports" / "raw_data_validation.json"
    )

    config = load_yaml(config_path)
    checks: list[Check] = []

    water_config = config["sources"]["water_demand"]
    weather_config = config["sources"]["noaa_weather"]

    water_directory = repository_root / water_config["output"]["directory"]
    water_metadata_path, water_metadata = find_latest_metadata(
        water_directory
    )
    water_csv_path = resolve_output_file(repository_root, water_metadata)
    water_dataframe = pd.read_csv(water_csv_path)

    configured_stations = weather_config["stations"]

    if len(configured_stations) != 1:
        raise ValueError(
            "This validation version expects exactly one configured NOAA "
            "station. Extend the script before adding multiple stations."
        )

    station_id = configured_stations[0]["id"]
    weather_directory = repository_root / weather_config["output"]["directory"]
    weather_metadata_path, weather_metadata = find_latest_metadata(
        weather_directory,
        station_id=station_id,
    )
    weather_csv_path = resolve_output_file(repository_root, weather_metadata)
    weather_dataframe = pd.read_csv(weather_csv_path)

    water_summary = validate_water_demand(
        dataframe=water_dataframe,
        source_config=water_config,
        metadata=water_metadata,
        csv_path=water_csv_path,
        checks=checks,
    )

    weather_summary = validate_noaa_weather(
        dataframe=weather_dataframe,
        source_config=weather_config,
        metadata=weather_metadata,
        csv_path=weather_csv_path,
        station_id=station_id,
        checks=checks,
    )

    water_date_column = water_config["fields"]["date"]["source_name"]
    water_dates = pd.to_datetime(
        water_dataframe[water_date_column],
        errors="coerce",
    ).dropna().dt.normalize()
    weather_dates = pd.to_datetime(
        weather_dataframe["DATE"],
        errors="coerce",
    ).dropna().dt.normalize()

    cross_source_summary = validate_cross_source_dates(
        water_dates=water_dates,
        weather_dates=weather_dates,
        checks=checks,
    )

    error_count = sum(check["status"] == "error" for check in checks)
    warning_count = sum(check["status"] == "warning" for check in checks)

    if error_count:
        overall_status = "fail"
    elif warning_count:
        overall_status = "pass_with_warnings"
    else:
        overall_status = "pass"

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config_path.relative_to(
            repository_root
        ).as_posix(),
        "status": overall_status,
        "summary": {
            "check_count": len(checks),
            "error_count": error_count,
            "warning_count": warning_count,
        },
        "inputs": {
            "water_metadata": water_metadata_path.relative_to(
                repository_root
            ).as_posix(),
            "water_csv": water_csv_path.relative_to(
                repository_root
            ).as_posix(),
            "weather_metadata": weather_metadata_path.relative_to(
                repository_root
            ).as_posix(),
            "weather_csv": weather_csv_path.relative_to(
                repository_root
            ).as_posix(),
        },
        "datasets": {
            "water_demand": water_summary,
            "noaa_weather": weather_summary,
            "cross_source": cross_source_summary,
        },
        "checks": checks,
    }

    write_json_atomically(output_path, report)

    print("Raw-data validation completed.")
    print(f"Status: {overall_status}")
    print(f"Checks: {len(checks)}")
    print(f"Errors: {error_count}")
    print(f"Warnings: {warning_count}")
    print(f"Report: {output_path}")

    if warning_count:
        print("\nWarnings:")
        for check in checks:
            if check["status"] == "warning":
                print(f"- {check['name']}: {check['message']}")

    if error_count:
        print("\nErrors:")
        for check in checks:
            if check["status"] == "error":
                print(f"- {check['name']}: {check['message']}")

        raise SystemExit(1)


if __name__ == "__main__":
    main()
