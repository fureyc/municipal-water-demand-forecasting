"""Time-aware train, validation and holdout splits for daily forecasting.

The functions in this module read the date boundaries in
``config/modeling.yml`` and convert them into positional row indices that can
be used with pandas or scikit-learn estimators.

The primary model-selection design is an expanding training window followed by
fixed-length rolling validation windows. The configured final holdout is kept
separate from every model-selection split.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class ExpandingWindowSplit:
    """One expanding-window train-validation split."""

    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    validation_start: pd.Timestamp
    validation_end: pd.Timestamp
    train_positions: np.ndarray
    validation_positions: np.ndarray

    @property
    def n_train(self) -> int:
        """Number of observations in the training window."""
        return int(self.train_positions.size)

    @property
    def n_validation(self) -> int:
        """Number of observations in the validation window."""
        return int(self.validation_positions.size)


@dataclass(frozen=True)
class HoldoutSplit:
    """The final untouched holdout period."""

    start: pd.Timestamp
    end: pd.Timestamp
    positions: np.ndarray

    @property
    def n_observations(self) -> int:
        """Number of observations in the holdout period."""
        return int(self.positions.size)


def load_modeling_config(
    config_path: str | Path = "config/modeling.yml",
) -> dict[str, Any]:
    """Load the modeling configuration from YAML.

    Parameters
    ----------
    config_path:
        Path to the modeling YAML file.

    Returns
    -------
    dict
        Parsed configuration.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the YAML file is empty or does not contain the required sections.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Modeling configuration not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Modeling configuration is empty or invalid: {path}")

    required_sections = {"forecast", "evaluation"}
    missing = required_sections.difference(config)

    if missing:
        raise ValueError(
            "Modeling configuration is missing required section(s): "
            + ", ".join(sorted(missing))
        )

    return config


def _timestamp(value: Any, field_name: str) -> pd.Timestamp:
    """Convert one configured date to a normalized pandas Timestamp."""
    try:
        result = pd.Timestamp(value).normalize()
    except Exception as exc:
        raise ValueError(f"Invalid date for {field_name}: {value!r}") from exc

    if pd.isna(result):
        raise ValueError(f"Invalid date for {field_name}: {value!r}")

    return result


def _prepare_dates(
    data: pd.DataFrame,
    date_column: str,
) -> pd.Series:
    """Validate and return the ordered daily date column."""
    if date_column not in data.columns:
        raise KeyError(f"Date column not found: {date_column}")

    dates = pd.to_datetime(data[date_column], errors="coerce").dt.normalize()

    if dates.isna().any():
        bad_rows = dates[dates.isna()].index.tolist()[:5]
        raise ValueError(
            f"Date column contains missing or invalid values near rows: {bad_rows}"
        )

    if dates.duplicated().any():
        duplicates = dates[dates.duplicated(keep=False)].unique()[:5]
        duplicate_text = ", ".join(pd.Timestamp(x).date().isoformat() for x in duplicates)
        raise ValueError(f"Date column contains duplicate dates: {duplicate_text}")

    if not dates.is_monotonic_increasing:
        raise ValueError(
            "Data must be sorted in increasing chronological order before "
            "time splits are generated."
        )

    expected_dates = pd.date_range(dates.iloc[0], dates.iloc[-1], freq="D")

    if len(expected_dates) != len(dates) or not np.array_equal(
        dates.to_numpy(), expected_dates.to_numpy()
    ):
        observed = pd.DatetimeIndex(dates)
        missing = expected_dates.difference(observed)

        preview = ", ".join(date.date().isoformat() for date in missing[:5])
        raise ValueError(
            "Date column is not a complete daily sequence."
            + (f" First missing date(s): {preview}" if preview else "")
        )

    return dates


def build_expanding_window_splits(
    data: pd.DataFrame,
    config: Mapping[str, Any],
) -> list[ExpandingWindowSplit]:
    """Construct expanding-window train-validation splits.

    Training always begins on the configured initial training start date.
    Before each validation window, the training endpoint expands to the final
    permissible date before that validation window.

    Parameters
    ----------
    data:
        Chronologically sorted daily dataframe.
    config:
        Parsed modeling configuration.

    Returns
    -------
    list of ExpandingWindowSplit
        Positional indices and date boundaries for each fold.
    """
    forecast = config["forecast"]
    evaluation = config["evaluation"]
    initial = evaluation["initial_training"]
    rolling = evaluation["rolling_validation"]
    holdout = evaluation["final_holdout"]

    date_column = forecast["date_column"]
    dates = _prepare_dates(data, date_column)

    training_start = _timestamp(
        initial["start_date"], "evaluation.initial_training.start_date"
    )
    configured_initial_end = _timestamp(
        initial["end_date"], "evaluation.initial_training.end_date"
    )
    validation_start = _timestamp(
        rolling["start_date"], "evaluation.rolling_validation.start_date"
    )
    validation_end = _timestamp(
        rolling["end_date"], "evaluation.rolling_validation.end_date"
    )
    holdout_start = _timestamp(
        holdout["start_date"], "evaluation.final_holdout.start_date"
    )

    window_months = int(rolling["window_months"])
    step_months = int(rolling["step_months"])
    gap_days = int(rolling.get("gap_days", 0))

    if window_months <= 0:
        raise ValueError("rolling_validation.window_months must be positive.")

    if step_months <= 0:
        raise ValueError("rolling_validation.step_months must be positive.")

    if gap_days < 0:
        raise ValueError("rolling_validation.gap_days cannot be negative.")

    expected_initial_end = validation_start - pd.Timedelta(days=gap_days + 1)

    if configured_initial_end != expected_initial_end:
        raise ValueError(
            "The configured initial training endpoint is inconsistent with "
            "the first validation date and gap. Expected "
            f"{expected_initial_end.date()}, found "
            f"{configured_initial_end.date()}."
        )

    if validation_end >= holdout_start:
        raise ValueError(
            "Rolling validation overlaps the configured final holdout."
        )

    if dates.iloc[0] > training_start:
        raise ValueError(
            "The dataset begins after the configured training start date."
        )

    if dates.iloc[-1] < validation_end:
        raise ValueError(
            "The dataset ends before the configured validation period."
        )

    splits: list[ExpandingWindowSplit] = []
    current_validation_start = validation_start
    fold = 1

    while current_validation_start <= validation_end:
        current_validation_end = min(
            current_validation_start + pd.DateOffset(months=window_months)
            - pd.Timedelta(days=1),
            validation_end,
        )
        current_train_end = current_validation_start - pd.Timedelta(
            days=gap_days + 1
        )

        train_mask = (
            (dates >= training_start)
            & (dates <= current_train_end)
        )
        validation_mask = (
            (dates >= current_validation_start)
            & (dates <= current_validation_end)
        )

        train_positions = np.flatnonzero(train_mask.to_numpy())
        validation_positions = np.flatnonzero(validation_mask.to_numpy())

        if train_positions.size == 0:
            raise ValueError(f"Fold {fold} has no training observations.")

        if validation_positions.size == 0:
            raise ValueError(f"Fold {fold} has no validation observations.")

        if np.intersect1d(train_positions, validation_positions).size:
            raise RuntimeError(f"Fold {fold} contains train-validation overlap.")

        if dates.iloc[validation_positions].max() >= holdout_start:
            raise RuntimeError(f"Fold {fold} includes final-holdout dates.")

        splits.append(
            ExpandingWindowSplit(
                fold=fold,
                train_start=training_start,
                train_end=current_train_end,
                validation_start=current_validation_start,
                validation_end=current_validation_end,
                train_positions=train_positions,
                validation_positions=validation_positions,
            )
        )

        fold += 1
        current_validation_start = (
            current_validation_start + pd.DateOffset(months=step_months)
        )

    if not splits:
        raise ValueError("No rolling-validation splits were generated.")

    return splits


def get_final_holdout(
    data: pd.DataFrame,
    config: Mapping[str, Any],
) -> HoldoutSplit:
    """Return the configured final holdout as positional indices."""
    forecast = config["forecast"]
    evaluation = config["evaluation"]
    holdout = evaluation["final_holdout"]

    date_column = forecast["date_column"]
    dates = _prepare_dates(data, date_column)

    holdout_start = _timestamp(
        holdout["start_date"], "evaluation.final_holdout.start_date"
    )
    holdout_end = _timestamp(
        holdout["end_date"], "evaluation.final_holdout.end_date"
    )

    if holdout_start > holdout_end:
        raise ValueError("Final holdout start date occurs after its end date.")

    mask = (dates >= holdout_start) & (dates <= holdout_end)
    positions = np.flatnonzero(mask.to_numpy())

    if positions.size == 0:
        raise ValueError("The configured final holdout contains no observations.")

    observed_start = dates.iloc[positions[0]]
    observed_end = dates.iloc[positions[-1]]

    if observed_start != holdout_start or observed_end != holdout_end:
        raise ValueError(
            "The dataset does not fully cover the configured final holdout."
        )

    return HoldoutSplit(
        start=holdout_start,
        end=holdout_end,
        positions=positions,
    )


def summarize_splits(
    splits: Sequence[ExpandingWindowSplit],
) -> pd.DataFrame:
    """Create a compact dataframe describing all validation folds."""
    rows = [
        {
            "fold": split.fold,
            "train_start": split.train_start.date().isoformat(),
            "train_end": split.train_end.date().isoformat(),
            "n_train": split.n_train,
            "validation_start": split.validation_start.date().isoformat(),
            "validation_end": split.validation_end.date().isoformat(),
            "n_validation": split.n_validation,
        }
        for split in splits
    ]

    return pd.DataFrame(rows)


def validate_holdout_isolation(
    splits: Sequence[ExpandingWindowSplit],
    holdout: HoldoutSplit,
) -> None:
    """Verify that no training or validation indices enter the holdout."""
    holdout_positions = holdout.positions

    for split in splits:
        train_overlap = np.intersect1d(
            split.train_positions, holdout_positions
        )
        validation_overlap = np.intersect1d(
            split.validation_positions, holdout_positions
        )

        if train_overlap.size or validation_overlap.size:
            raise RuntimeError(
                f"Fold {split.fold} overlaps the final holdout."
            )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect configured expanding-window time splits."
    )
    parser.add_argument(
        "--data",
        default="data/processed/fort_collins_daily_water_weather.csv",
        help="Path to the processed daily dataset.",
    )
    parser.add_argument(
        "--config",
        default="config/modeling.yml",
        help="Path to the modeling configuration.",
    )
    return parser


def main() -> None:
    """Command-line entry point for validating and displaying the splits."""
    parser = _build_argument_parser()
    args = parser.parse_args()

    config = load_modeling_config(args.config)
    date_column = config["forecast"]["date_column"]

    data = pd.read_csv(args.data, parse_dates=[date_column])

    splits = build_expanding_window_splits(data, config)
    holdout = get_final_holdout(data, config)
    validate_holdout_isolation(splits, holdout)

    summary = summarize_splits(splits)

    print(summary.to_string(index=False))
    print()
    print(f"Validation folds: {len(splits)}")
    print(
        "Final holdout: "
        f"{holdout.start.date()} through {holdout.end.date()} "
        f"({holdout.n_observations} observations)"
    )
    print("Holdout isolation check: passed")


if __name__ == "__main__":
    main()
