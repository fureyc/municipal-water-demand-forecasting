# Fort Collins Municipal Water-Demand Forecasting

A reproducible time-series project combining Fort Collins daily water-demand
data with NOAA weather observations to study short-term demand forecasting and
weather normalization.

## Project overview

I created this project as a compact demonstration of how I would approach a
municipal water-demand forecasting problem using public data. The workflow
begins with programmatic data acquisition and validation, then combines daily
demand with temperature, precipitation and snow observations.

The modeling stage will preserve chronological order and begin with naïve,
seasonal and linear baselines. More flexible models will only be introduced
when they show a consistent improvement under the same time-aware evaluation
procedure.

This repository is intended as an analytical portfolio project rather than an
operational forecasting or utility-planning system.

## Current status

The following stages are complete:

- water-demand data acquisition
- NOAA weather-data acquisition
- raw-data validation
- construction of a complete daily modeling table
- reproducible data audit
- documentation of methodological decisions and project references

Current work focuses on:

- exploratory analysis of lagged and seasonal structure
- supervised forecasting-feature construction
- expanding-window model evaluation
- naïve, seasonal and regression baselines

## Data

The processed dataset contains 2,282 complete daily observations from
January 1, 2020 through March 31, 2026.

- **Forecast target:** observed daily municipal water demand in millions of
  gallons per day
- **Demand source:** City of Fort Collins Daily Water Demand
- **Weather source:** NOAA Daily Summaries
- **Primary weather station:** `USC00053005`, FORT COLLINS, CO US
- **Core weather variables:** daily maximum temperature, minimum temperature,
  precipitation, snowfall and snow depth

Average wind speed is preserved in the raw NOAA snapshot but excluded from the
initial processed table because it is the only requested weather variable with
substantial missingness.

The City's projected-demand field is retained as a comparison benchmark rather
than used as a default predictor. The observed-to-projected ratio is also
excluded from predictors because it directly contains the forecasting target.

## Methodological approach

The observations are treated as an ordered daily time series rather than as
exchangeable tabular records. Random train-test splits will not be used for the
primary evaluation.

Model comparison and tuning will use an expanding-window rolling-origin
design. A final period will remain untouched during model selection and will
be used for the final reported comparison.

Lagged demand variables and rolling summaries will be constructed using only
information available before each forecast date.

The initial model progression is:

1. naïve and seasonal forecasts
2. calendar-only regression
3. weather and calendar regression
4. lagged-demand regression
5. more flexible models when justified by validation performance

Simple baselines are an important part of the project rather than a formality.
Time-series forecasting is difficult and well-chosen seasonal or naïve methods
can be genuinely hard to improve upon.

## Key project artifacts

- [Rendered data audit](reports/data_audit.html)
- [Reproducible data-audit source](reports/data_audit.qmd)
- [Raw-data validation report](reports/raw_data_validation.json)
- [Methodological decisions](docs/methodological_decisions.md)
- [Project references](docs/references.bib)
- [Data-source configuration](config/data_sources.yml)

## Repository structure

```text
municipal-water-demand-forecasting/
├── config/
│   └── data_sources.yml
├── data/
│   ├── raw/
│   │   ├── noaa/
│   │   └── water_demand/
│   └── processed/
├── docs/
│   ├── methodological_decisions.md
│   └── references.bib
├── reports/
│   ├── data_audit.qmd
│   ├── data_audit.html
│   └── raw_data_validation.json
├── src/
│   └── data/
│       ├── download_water_demand.py
│       ├── download_noaa_weather.py
│       ├── validate_raw_data.py
│       └── build_daily_dataset.py
├── .gitignore
├── README.md
└── requirements.txt
