
"""Leakage-safe feature construction for daily water-demand forecasting.

This module builds the two operational feature matrices finalized during the
exploratory analysis:

- Matrix A: a curated 27-feature specification
- Matrix B: a broader 54-feature specification

Features are constructed on the complete ordered daily series before any
chronological split is applied. Lagged and rolling demand features use only
observations available before the forecast date.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar


DATE_COLUMN = "date"
TARGET_COLUMN = "water_demand_mgd"

REFERENCE_COLUMNS = (
    "projected_water_demand_mgd",
    "actual_projected_demand_ratio",
    "plant_demand_mgd",
)

REQUIRED_SOURCE_COLUMNS = (
    DATE_COLUMN,
    TARGET_COLUMN,
    "tavg_f",
    "temperature_range_f",
    "precipitation_in",
    "snowfall_in",
    "snow_depth_in",
)


MONTH_FEATURES = (
    "month_Feb",
    "month_Mar",
    "month_Apr",
    "month_May",
    "month_Jun",
    "month_Jul",
    "month_Aug",
    "month_Sep",
    "month_Oct",
    "month_Nov",
    "month_Dec",
)

WEEKDAY_FEATURES = (
    "dow_Tue",
    "dow_Wed",
    "dow_Thu",
    "dow_Fri",
    "dow_Sat",
    "dow_Sun",
)

HOLIDAY_FEATURES = (
    "is_observed_federal_holiday",
    "is_day_before_observed_federal_holiday",
    "is_day_after_observed_federal_holiday",
)

ANNUAL_HARMONIC_FEATURES = (
    "annual_sin_1",
    "annual_cos_1",
    "annual_sin_2",
    "annual_cos_2",
)

MATRIX_A_DEMAND_FEATURES = (
    "demand_lag_1",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
)

MATRIX_B_DIRECT_DEMAND_FEATURES = (
    "demand_lag_1",
    "demand_lag_2",
    "demand_lag_3",
    "demand_lag_7",
    "demand_lag_14",
    "demand_lag_28",
)

ROLLING_DEMAND_LEVEL_FEATURES = (
    "demand_roll_mean_7",
    "demand_roll_mean_14",
    "demand_roll_mean_28",
)

ROLLING_DEMAND_VARIABILITY_FEATURES = (
    "demand_roll_std_7",
    "demand_roll_std_14",
    "demand_roll_std_28",
)

LAGGED_TEMPERATURE_FEATURES = (
    "tavg_f_lag_1",
    "tavg_f_lag_2",
    "tavg_f_lag_7",
    "temperature_range_f_lag_1",
    "temperature_range_f_lag_2",
    "temperature_range_f_lag_7",
)

LAGGED_PRECIPITATION_FEATURES = (
    "precipitation_in_lag_1",
    "precipitation_in_lag_2",
    "precipitation_in_lag_7",
    "has_precipitation_lag_1",
    "has_precipitation_lag_2",
    "has_precipitation_lag_7",
)

LAGGED_SNOW_FEATURES = (
    "snowfall_in_lag_1",
    "snowfall_in_lag_2",
    "snowfall_in_lag_7",
    "snow_depth_in_lag_1",
    "snow_depth_in_lag_2",
    "snow_depth_in_lag_7",
)


MATRIX_A_GROUPS = {
    "annual_calendar": MONTH_FEATURES,
    "weekly_calendar": WEEKDAY_FEATURES,
    "holiday_calendar": HOLIDAY_FEATURES,
    "demand_history": MATRIX_A_DEMAND_FEATURES,
    "lagged_temperature": (
        "tavg_f_lag_1",
    ),
    "lagged_precipitation": (
        "precipitation_in_lag_1",
        "precipitation_in_lag_2",
    ),
}

MATRIX_B_GROUPS = {
    "annual_calendar": (
        MONTH_FEATURES
        + ANNUAL_HARMONIC_FEATURES
    ),
    "weekly_calendar": WEEKDAY_FEATURES,
    "holiday_calendar": HOLIDAY_FEATURES,
    "direct_demand_lags": (
        MATRIX_B_DIRECT_DEMAND_FEATURES
    ),
    "rolling_demand_level": (
        ROLLING_DEMAND_LEVEL_FEATURES
    ),
    "rolling_demand_variability": (
        ROLLING_DEMAND_VARIABILITY_FEATURES
    ),
    "lagged_temperature": (
        LAGGED_TEMPERATURE_FEATURES
    ),
    "lagged_precipitation": (
        LAGGED_PRECIPITATION_FEATURES
    ),
    "lagged_snow": LAGGED_SNOW_FEATURES,
}


MATRIX_A_SAME_DAY_WEATHER_ADDONS = (
    "same_day_tavg_f",
    "same_day_precipitation_in",
)

MATRIX_B_SAME_DAY_WEATHER_ADDONS = (
    "same_day_tavg_f",
    "same_day_temperature_range_f",
    "same_day_precipitation_in",
    "same_day_has_precipitation",
)


def _flatten_groups(
    groups: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Flatten ordered feature groups into one ordered tuple."""
    return tuple(
        feature
        for features in groups.values()
        for feature in features
    )


MATRIX_A_FEATURES = _flatten_groups(
    MATRIX_A_GROUPS
)

MATRIX_B_FEATURES = _flatten_groups(
    MATRIX_B_GROUPS
)


assert len(MATRIX_A_FEATURES) == 27
assert len(MATRIX_B_FEATURES) == 54
assert len(set(MATRIX_A_FEATURES)) == 27
assert len(set(MATRIX_B_FEATURES)) == 54
assert set(MATRIX_A_FEATURES).issubset(
    set(MATRIX_B_FEATURES)
)


@dataclass(frozen=True)
class ForecastingFeatureSet:
    """Container for aligned forecasting features and metadata."""

    table: pd.DataFrame
    dates: pd.Series
    target: pd.Series
    matrix_a: pd.DataFrame
    matrix_b: pd.DataFrame
    matrix_a_weather_informed: pd.DataFrame
    matrix_b_weather_informed: pd.DataFrame
    matrix_a_groups: Mapping[
        str,
        tuple[str, ...],
    ]
    matrix_b_groups: Mapping[
        str,
        tuple[str, ...],
    ]
    matrix_a_same_day_weather_addons: tuple[
        str,
        ...,
    ]
    matrix_b_same_day_weather_addons: tuple[
        str,
        ...,
    ]


def load_processed_data(
    path: str | Path,
) -> pd.DataFrame:
    """Load the processed daily water-and-weather dataset."""
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {path}"
        )

    data = pd.read_csv(path)

    if DATE_COLUMN not in data.columns:
        raise ValueError(
            f"Dataset must contain '{DATE_COLUMN}'."
        )

    data[DATE_COLUMN] = pd.to_datetime(
        data[DATE_COLUMN],
        errors="raise",
    )

    return data


def _validate_source_data(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Validate ordering, daily continuity and required values."""
    missing_columns = sorted(
        set(REQUIRED_SOURCE_COLUMNS)
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Missing required source columns: "
            + ", ".join(missing_columns)
        )

    validated = data.copy()

    validated[DATE_COLUMN] = pd.to_datetime(
        validated[DATE_COLUMN],
        errors="raise",
    )

    validated = (
        validated
        .sort_values(DATE_COLUMN)
        .reset_index(drop=True)
    )

    if validated[DATE_COLUMN].duplicated().any():
        duplicated_dates = (
            validated.loc[
                validated[DATE_COLUMN].duplicated(),
                DATE_COLUMN,
            ]
            .dt.strftime("%Y-%m-%d")
            .tolist()
        )

        raise ValueError(
            "Duplicate dates found: "
            + ", ".join(duplicated_dates[:10])
        )

    expected_dates = pd.date_range(
        start=validated[DATE_COLUMN].min(),
        end=validated[DATE_COLUMN].max(),
        freq="D",
    )

    observed_dates = pd.DatetimeIndex(
        validated[DATE_COLUMN]
    )

    if not observed_dates.equals(expected_dates):
        missing_dates = expected_dates.difference(
            observed_dates
        )

        raise ValueError(
            "The processed series is not complete and daily. "
            f"Missing dates include: "
            f"{missing_dates[:10].tolist()}"
        )

    source_missing_counts = (
        validated[
            list(REQUIRED_SOURCE_COLUMNS[1:])
        ]
        .isna()
        .sum()
    )

    source_missing_counts = (
        source_missing_counts.loc[
            source_missing_counts > 0
        ]
    )

    if not source_missing_counts.empty:
        raise ValueError(
            "Missing values found in required source columns: "
            + source_missing_counts.to_dict().__repr__()
        )

    return validated


def _add_calendar_features(
    table: pd.DataFrame,
) -> None:
    """Add month, weekday, holiday and harmonic features."""
    dates = table[DATE_COLUMN]

    month_names = {
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }

    for month_number, month_name in (
        month_names.items()
    ):
        table[f"month_{month_name}"] = (
            dates.dt.month.eq(month_number)
            .astype("int8")
        )

    weekday_names = {
        1: "Tue",
        2: "Wed",
        3: "Thu",
        4: "Fri",
        5: "Sat",
        6: "Sun",
    }

    for weekday_number, weekday_name in (
        weekday_names.items()
    ):
        table[f"dow_{weekday_name}"] = (
            dates.dt.dayofweek.eq(
                weekday_number
            )
            .astype("int8")
        )

    holiday_calendar = (
        USFederalHolidayCalendar()
    )

    observed_holidays = pd.DatetimeIndex(
        holiday_calendar.holidays(
            start=dates.min()
            - pd.Timedelta(days=1),
            end=dates.max()
            + pd.Timedelta(days=1),
        )
    )

    table[
        "is_observed_federal_holiday"
    ] = (
        dates.isin(observed_holidays)
        .astype("int8")
    )

    table[
        "is_day_before_observed_federal_holiday"
    ] = (
        (
            dates
            + pd.Timedelta(days=1)
        )
        .isin(observed_holidays)
        .astype("int8")
    )

    table[
        "is_day_after_observed_federal_holiday"
    ] = (
        (
            dates
            - pd.Timedelta(days=1)
        )
        .isin(observed_holidays)
        .astype("int8")
    )

    annual_phase = (
        2.0
        * np.pi
        * dates.dt.dayofyear
        / 365.25
    )

    for harmonic in (1, 2):
        table[
            f"annual_sin_{harmonic}"
        ] = np.sin(
            harmonic * annual_phase
        )

        table[
            f"annual_cos_{harmonic}"
        ] = np.cos(
            harmonic * annual_phase
        )


def _add_demand_history_features(
    table: pd.DataFrame,
) -> None:
    """Add leakage-safe demand lags and rolling summaries."""
    target = table[TARGET_COLUMN]

    for lag in (1, 2, 3, 7, 14, 28):
        table[f"demand_lag_{lag}"] = (
            target.shift(lag)
        )

    shifted_target = target.shift(1)

    for window in (7, 14, 28):
        rolling = shifted_target.rolling(
            window=window,
            min_periods=window,
        )

        table[
            f"demand_roll_mean_{window}"
        ] = rolling.mean()

        table[
            f"demand_roll_std_{window}"
        ] = rolling.std()


def _add_weather_features(
    table: pd.DataFrame,
) -> None:
    """Add lagged operational weather and same-day benchmarks."""
    has_precipitation = (
        table["precipitation_in"] > 0
    ).astype("int8")

    for lag in (1, 2, 7):
        table[f"tavg_f_lag_{lag}"] = (
            table["tavg_f"].shift(lag)
        )

        table[
            f"temperature_range_f_lag_{lag}"
        ] = (
            table["temperature_range_f"]
            .shift(lag)
        )

        table[
            f"precipitation_in_lag_{lag}"
        ] = (
            table["precipitation_in"]
            .shift(lag)
        )

        table[
            f"has_precipitation_lag_{lag}"
        ] = has_precipitation.shift(lag)

        table[
            f"snowfall_in_lag_{lag}"
        ] = (
            table["snowfall_in"].shift(lag)
        )

        table[
            f"snow_depth_in_lag_{lag}"
        ] = (
            table["snow_depth_in"].shift(lag)
        )

    table["same_day_tavg_f"] = (
        table["tavg_f"]
    )

    table[
        "same_day_temperature_range_f"
    ] = table["temperature_range_f"]

    table[
        "same_day_precipitation_in"
    ] = table["precipitation_in"]

    table[
        "same_day_has_precipitation"
    ] = has_precipitation


def build_forecasting_features(
    data: pd.DataFrame,
) -> ForecastingFeatureSet:
    """Build aligned operational and weather-informed matrices."""
    source = _validate_source_data(data)

    retained_columns = [
        DATE_COLUMN,
        TARGET_COLUMN,
    ]

    retained_columns.extend(
        column
        for column in REFERENCE_COLUMNS
        if column in source.columns
    )

    retained_columns.extend(
        [
            "tavg_f",
            "temperature_range_f",
            "precipitation_in",
            "snowfall_in",
            "snow_depth_in",
        ]
    )

    table = source[
        retained_columns
    ].copy()

    _add_calendar_features(table)
    _add_demand_history_features(table)
    _add_weather_features(table)

    alignment_columns = (
        list(MATRIX_B_FEATURES)
        + [TARGET_COLUMN]
    )

    complete_row_mask = (
        table[alignment_columns]
        .notna()
        .all(axis=1)
    )

    table = (
        table.loc[complete_row_mask]
        .reset_index(drop=True)
    )

    matrix_a = table[
        list(MATRIX_A_FEATURES)
    ].astype(float)

    matrix_b = table[
        list(MATRIX_B_FEATURES)
    ].astype(float)

    matrix_a_weather_informed = table[
        list(MATRIX_A_FEATURES)
        + list(
            MATRIX_A_SAME_DAY_WEATHER_ADDONS
        )
    ].astype(float)

    matrix_b_weather_informed = table[
        list(MATRIX_B_FEATURES)
        + list(
            MATRIX_B_SAME_DAY_WEATHER_ADDONS
        )
    ].astype(float)

    target = (
        table[TARGET_COLUMN]
        .astype(float)
        .rename(TARGET_COLUMN)
    )

    dates = (
        table[DATE_COLUMN]
        .copy()
        .rename(DATE_COLUMN)
    )

    matrices = {
        "matrix_a": matrix_a,
        "matrix_b": matrix_b,
        "matrix_a_weather_informed": (
            matrix_a_weather_informed
        ),
        "matrix_b_weather_informed": (
            matrix_b_weather_informed
        ),
    }

    for matrix_name, matrix in matrices.items():
        if matrix.isna().any().any():
            raise ValueError(
                f"{matrix_name} contains missing values."
            )

        if not np.isfinite(
            matrix.to_numpy(dtype=float)
        ).all():
            raise ValueError(
                f"{matrix_name} contains non-finite values."
            )

        if len(matrix) != len(target):
            raise ValueError(
                f"{matrix_name} is not aligned "
                "with the target."
            )

    return ForecastingFeatureSet(
        table=table,
        dates=dates,
        target=target,
        matrix_a=matrix_a,
        matrix_b=matrix_b,
        matrix_a_weather_informed=(
            matrix_a_weather_informed
        ),
        matrix_b_weather_informed=(
            matrix_b_weather_informed
        ),
        matrix_a_groups=MATRIX_A_GROUPS,
        matrix_b_groups=MATRIX_B_GROUPS,
        matrix_a_same_day_weather_addons=(
            MATRIX_A_SAME_DAY_WEATHER_ADDONS
        ),
        matrix_b_same_day_weather_addons=(
            MATRIX_B_SAME_DAY_WEATHER_ADDONS
        ),
    )


def load_and_build_forecasting_features(
    path: str | Path,
) -> ForecastingFeatureSet:
    """Load the processed dataset and build all feature matrices."""
    return build_forecasting_features(
        load_processed_data(path)
    )
