# Methodological Decisions

## Project scope

This project is intended as a compact and reproducible demonstration of municipal water-demand forecasting. The goal is not to build a production forecasting system or make operational recommendations for the City of Fort Collins. Instead, I want to show how I would approach a time-series problem that combines public utility data with weather observations, beginning with careful data validation and moving toward a small set of interpretable forecasting models.

## Data-source selection

I selected the Fort Collins daily water-demand dataset because it provides a public daily series that can be accessed programmatically. The daily frequency makes it suitable for studying short-term demand patterns and for combining the demand series with daily weather observations.

Weather data are drawn from the NOAA Daily Summaries dataset. I selected station `USC00053005`, FORT COLLINS, CO US, as the primary station because it is located in Fort Collins and provides complete coverage for the core weather variables over the analysis period.

The analysis covers January 1, 2020 through March 31, 2026. This is the full common period supported by the acquired demand and weather snapshots. The 2026 observations represent only a partial calendar year, which will be taken into account when presenting annual summaries or comparing results across years.

## Forecast target and reference fields

The primary target is `water_demand_mgd`. It represents the observed quantity this project is intended to forecast: daily municipal water demand measured in millions of gallons per day.

The remaining demand-related variables are retained for reference rather than used as default predictors.

`projected_water_demand_mgd` contains the City’s existing daily projection. Rather than using it as an input feature, I plan to treat it as a useful benchmark for comparison with the models developed in this project.

`actual_projected_demand_ratio` is calculated using the observed target. Including it as a predictor would therefore introduce direct target leakage.

`plant_demand_mgd` is retained for context, but I have not established that it represents the same demand concept as the primary target or that it would be available when a forecast is made. It will not be used as a predictor unless its interpretation and forecast-time availability can be justified.

## Weather variables

The initial processed dataset includes daily maximum temperature, minimum temperature, precipitation, snowfall and snow depth. These variables have complete coverage for the selected station and analysis period.

Average wind speed, `AWND`, is preserved in the raw NOAA snapshot but excluded from the first processed dataset because it is the only requested weather variable with substantial missingness. For a small demonstration project, I preferred to begin with a complete analysis table rather than introduce an imputation method whose effect would need to be separately evaluated. Wind speed could be reconsidered later if it appears likely to add enough predictive value to justify additional preprocessing.

## Time-aware evaluation

Because these observations form a daily time series, the order of the rows is meaningful. Unlike an ordinary tabular regression problem, the data should not be treated as independent observations that can be randomly divided into training and test sets.

A random split could allow later observations to influence models evaluated on earlier dates. It could also make performance appear stronger by placing nearby observations with similar seasonal and demand patterns on both sides of the split.

Model comparison and tuning will use an expanding-window rolling-origin design. Each training period will contain all observations available before a forecast origin, then the forecast origin will move forward to create another evaluation period. This provides several chronological evaluations while reflecting the way a model would be updated as new demand observations become available.

A final period will remain untouched during model selection and tuning. This holdout period will be used for the final reported comparison.

Lagged demand variables and rolling summaries will also be created using only information available before the date being predicted. Any rolling calculation will be shifted as needed to prevent the current or future target from entering the feature set.

## Baseline-first modeling

I will begin with simple naïve, seasonal and linear baselines before considering more flexible models. These may include previous-day demand, demand from the same day one week earlier, seasonal averages and linear regression models based on calendar and weather variables.

This is not only a matter of interpretability. Time-series forecasting is difficult and in my experience well-chosen baselines can be genuinely hard to improve upon. Beginning with simple models provides a meaningful performance floor and helps determine whether additional complexity is actually useful. 
## Numerical considerations

Lagged demand variables, rolling summaries, seasonal encodings and related weather features may create substantial correlation within the design matrix. My research background is in numerical linear algebra, so I will monitor conditioning and redundancy as the supervised feature set is constructed.

The current dataset is small enough that standard, numerically stable implementations should be sufficient. I do not plan to introduce specialized numerical linear algebra techniques unless the scale or structure of the modeling problem provides a clear reason to do so. For linear models, I will also avoid manually forming and solving the normal equations when established QR- or SVD-based implementations provide a more stable alternative.

## Known limitations

This project uses one municipal demand series and one primary weather station. A single station cannot capture every local weather pattern across a utility service area.

The final year contains data only through March 31, 2026.

The weather variables are observed historical values. A deployable forecasting system would need to account for the availability and uncertainty of weather forecasts.

Fort Collins’ open-data platform was undergoing a migration I acquired the water-demand data (August 3rd). The project therefore preserves dated raw snapshots and acquisition metadata so that future updates can be compared against the current files.

These limitations are consistent with the intended scope of the project. The results should be viewed as an analytical demonstration rather than an operational water-demand planning tool.
