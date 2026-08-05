# Municipal Water Demand Forecasting

A reproducible one-day-ahead forecasting project that combines Fort Collins
water-demand data with NOAA weather observations.

The final XGBoost model achieved a holdout mean absolute error of **0.876 million
gallons per day**, improving on previous-day persistence by **31.8%** and ridge
regression by **11.8%**.

## Overview

Municipal water demand follows strong seasonal patterns, but it can also change
quickly in response to weather, calendar effects and recent consumption. This
project asks a practical forecasting question:

> Can tomorrow's municipal water demand be predicted more accurately using
> recent demand, calendar information and lagged weather than by simply using
> today's demand?

I developed the project as an application of statistical modeling and
time-series forecasting to a water-management problem. I was especially
interested in understanding not only whether a more flexible model could
improve forecast accuracy, but when it helped and where it still failed.

The project uses public data, chronological validation and a final untouched
holdout period. Feature choices and model settings were fixed before the
holdout results were viewed.

## Why I built this project

My academic work has focused on machine learning, uncertainty quantification
and scalable numerical methods. I built this project to explore how those
skills could be applied to a practical conservation and utility-planning
problem.

Public water-system reports often distinguish between observed demand and
weather-normalized demand. That raised several related questions for me:

- How much of tomorrow's demand can be predicted from recent consumption?
- Does lagged weather provide information beyond demand history and seasonality?
- Do linear and nonlinear models use that information differently?
- Under what conditions does a forecasting model fail?

Rather than begin with a preferred method, I organized the project around a
transparent comparison of simple baselines, linear models and a nonlinear
tree-based model.

## Forecasting task

The target is daily municipal water demand measured in millions of gallons per
day.

The forecasting horizon is one day:

$$
\widehat{y}_{t+1}
=
f(\text{information available through day } t).
$$

The operational feature set includes only information that would be available
by the end of the previous day. Same-day observed weather is evaluated only as
a retrospective benchmark and is not used by the final operational model.

## Data

The processed dataset combines:

- daily Fort Collins water demand from the City of Fort Collins
- daily weather observations from NOAA station `USC00053005`
- calendar and holiday information derived from the forecast date

The processed data cover **January 1, 2020 through March 31, 2026** and contain
**2,282 complete daily observations**.

Weather variables include temperature, precipitation and snow. Lagged demand
features include recent daily values, rolling means and rolling variability.

The source data are downloaded and processed through scripts in `src/data/`.
The resulting modeling dataset is stored at:

```text
data/processed/fort_collins_daily_water_weather.csv
