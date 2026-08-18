# Methodological Decisions

## Project scope

This project is intended as a compact and reproducible demonstration of municipal
water-demand forecasting. The goal is not to build a production forecasting system
or make operational recommendations for the City of Fort Collins. Instead, I use the
project to show how I would approach a time-series problem that combines public utility
data with weather observations, beginning with data validation and chronological
evaluation, progressing through point forecasting, and then extending the analysis to
probabilistic forecasting and uncertainty calibration.

The emphasis is on transparent methodological choices, simple benchmarks, reproducible
evaluation and clear separation between exploratory analysis and final reported results.
Where a more complicated method was considered, I retained it only when the data provided
a clear reason to do so.

## Data-source selection

I selected the Fort Collins daily water-demand dataset because it provides a public daily
series that can be accessed programmatically. The daily frequency makes it suitable for
studying short-term demand patterns and for combining the demand series with daily weather
observations.

Weather data are drawn from the NOAA Daily Summaries dataset. I selected station
`USC00053005`, FORT COLLINS, CO US, as the primary station because it is located in
Fort Collins and provides complete coverage for the core weather variables over the
analysis period.

The analysis covers January 1, 2020 through March 31, 2026. This is the full common
period supported by the acquired demand and weather snapshots. Because the data end on
March 31, 2026, calendar-year summaries for 2026 should not be compared directly with
complete earlier years.

## Forecast target and reference fields

The primary target is `water_demand_mgd`. It represents the observed quantity this
project is intended to forecast: daily municipal water demand measured in millions of
gallons per day.

The remaining demand-related variables are retained for reference rather than used as
default predictors.

`projected_water_demand_mgd` contains the City's existing daily projection. I retain it
as a reference field rather than using it as a model input.

`actual_projected_demand_ratio` is calculated using the observed target. Including it
as a predictor would therefore introduce direct target leakage.

`plant_demand_mgd` is retained for context, but I did not establish that it represents
the same demand concept as the primary target or that it would be available when a
forecast is made. It is therefore excluded from the predictor set.

## Weather variables and forecast-time availability

The processed dataset includes daily maximum temperature, minimum temperature,
precipitation, snowfall and snow depth. These variables have complete coverage for the
selected station and analysis period.

Average wind speed, `AWND`, is preserved in the raw NOAA snapshot but excluded from the
processed modeling dataset because it is the requested weather variable with substantial
missingness. For a compact demonstration project, I preferred to begin with a complete
analysis table rather than introduce an imputation procedure whose effect would need to
be evaluated separately.

An important distinction is made between historical weather observations and information
that would actually be available when issuing a one-day-ahead forecast. Observed
same-day weather is not used in the operational feature set because it would not yet be
known at forecast time. The deployable feature matrix therefore uses lagged weather
variables.

A model using observed same-day weather was evaluated only as a retrospective benchmark.
Its purpose is to estimate how much additional predictive value might be available from
better weather information, not to represent a deployable forecasting procedure. Archived
one-day-ahead weather forecasts would be a natural extension of the project.

This information boundary is also preserved in the probabilistic interpretation.
Weather-regime analyses use prior-day observed conditions rather than same-day
observations.

## Time-aware evaluation

Because these observations form a daily time series, row order is meaningful. The data
are not treated as independent observations that can be randomly divided into training
and test sets.

A random split could allow later observations to influence models evaluated on earlier
dates. It could also make performance appear stronger by placing nearby observations
with similar seasonal and demand patterns on both sides of the split.

Point-model comparison and tuning therefore use an expanding-window rolling-origin
design. The primary validation period runs from January 2023 through March 2025 and
contains nine quarterly validation folds. In each fold, a model is fitted using all
observations available before the validation quarter and then evaluated on that quarter.
After evaluation, the quarter becomes part of the historical training record available
to the next fold.

The period from April 2025 through March 2026 was reserved as an untouched final holdout
while point-model features, model families and hyperparameters were selected. After those
decisions were frozen, the holdout was opened once for the final reported comparison. It
will not be reused for subsequent feature selection, hyperparameter tuning or model
comparison.

The later probabilistic-forecasting analysis is treated as new work rather than a
retroactive modification of the point-model comparison. It does not reuse the opened
holdout as a new untouched test set. For the sequential calibration analysis, 2021
provides an initial conformity-score history, 2022 is used for calibration-method and
parameter selection, and January 2023 through March 2025 is used for the reported
prospective comparison.

Lagged demand variables and rolling summaries are constructed using only information
available before the forecast date. Rolling calculations are shifted as needed so that
the current or future target cannot enter the feature set.

## Baseline-first model comparison

I began with previous-day persistence as the primary operational baseline. This is a
demanding benchmark because daily municipal water use has strong short-term dependence:
when demand changes little, yesterday's value can be difficult to improve upon.

Ridge regression was retained as the principal linear benchmark because the broader
feature matrix contains substantial correlation among demand lags, rolling summaries,
calendar variables and weather predictors.

XGBoost was then used to test whether nonlinear relationships, thresholds and feature
interactions produced meaningful additional predictive value.

The final point-model comparison therefore follows a deliberate progression from a
simple persistence forecast, to a regularized linear model, to a nonlinear
gradient-boosted tree model. XGBoost produced the lowest rolling-validation error and
was selected before the final holdout was opened.

The simpler models remain important even though they were not the final winner. Their
performance helps identify when additional model flexibility is useful and when a very
simple forecast remains difficult to beat.

## Feature-set decisions

Two related feature matrices were retained during model development.

**Matrix A** contains 27 curated predictors and was used primarily for early
linear-model diagnostics and matrix-geometry analysis.

**Matrix B** contains 54 predictors and includes a broader collection of recent demand
lags, rolling demand levels and variability, annual and weekly calendar terms, holiday
indicators and lagged weather variables.

The broader matrix is more correlated, particularly because direct demand lags and
rolling summaries contain overlapping information. It nevertheless produced better
predictive performance. I therefore retained Matrix B for the final Ridge, XGBoost and
probabilistic-forecasting analyses rather than removing predictors solely to improve
matrix conditioning.

All lagged and rolling predictors preserve the one-day-ahead information boundary.

## Numerical considerations

Lagged demand variables, rolling summaries, seasonal encodings and related weather
features create substantial correlation within the broader design matrix. Because my
research background is in numerical linear algebra, I examined conditioning and
redundancy explicitly rather than treating them only as modeling concerns.

After standardization, the curated Matrix A was well conditioned. Matrix B was more
correlated because several predictors contain overlapping information. Ridge
regularization improved coefficient stability while preserving the predictive benefit
of the broader feature set.

Principal component regression was also evaluated as a possible dimension-reduction
strategy. Truncating low-variance directions consistently reduced validation accuracy,
even when more than 99% of predictor variance was retained. Predictor variance alone was
therefore not a useful feature-selection criterion for this problem.

The dataset remains small enough that standard numerical implementations are sufficient.
I did not introduce specialized randomized or large-scale numerical linear algebra
methods where the size or structure of the problem did not provide a genuine
computational reason to do so.

## Probabilistic forecasting

After completing the point-forecasting analysis, I extended the project from predicting
a single value to estimating a distribution of plausible demand outcomes.

Linear quantile regression and quantile XGBoost were compared at the 0.10, 0.50 and
0.90 quantiles using the same chronological validation philosophy as the point-model
analysis.

Quantile XGBoost achieved a slightly lower average pinball loss across the three
quantiles. Linear quantile regression, however, produced better empirical quantile
calibration at all three probability levels and substantially lower upper-tail pinball
loss at the 0.90 quantile. The difference in average pinball loss was small, so I
retained Linear QR as a transparent and competitive probabilistic baseline rather than
selecting a model from a single aggregate metric.

The raw 10th-to-90th percentile Linear QR interval contained approximately 74.4% of
observations over the January 2023 through March 2025 evaluation period, below its
nominal 80% level. The width of the raw interval varied substantially through time,
indicating that the quantile model captured meaningful heteroskedasticity, but the
intervals still required calibration.

## Sequential uncertainty calibration

Ordinary conformal methods are most straightforward under exchangeability, while daily
water demand is a sequential time series whose error distribution can change through
time. I therefore focused on calibration procedures that update as forecast errors
become available.

The comparison included a rolling conformal baseline, two-sided Adaptive Conformal
Inference (ACI), two-sided Conformal PID and an asymmetric PID variant.

Calibration-memory lengths and controller parameters were selected using 2022 only.
The subsequent January 2023 through March 2025 period was then used for the reported
comparison.

The calibration methods were evaluated using more than pooled coverage alone. I also
considered average interval width, lower- and upper-tail miss rates and the stability of
90-day trailing empirical coverage. This was intended to avoid selecting a method that
achieved nominal average coverage only by producing unnecessarily wide or locally
unstable intervals.

Two-sided ACI with a 180-day conformity-score history was retained as the preferred
method. Over the 2023 through March 2025 evaluation period, it increased empirical
coverage from approximately 74.4% for the raw Linear QR interval to approximately
80.6%, while average interval width increased from about 2.58 to 2.88 MGD. Lower- and
upper-tail miss rates were both close to 10%.

Other calibration methods were also competitive. The rolling conformal baseline
produced slightly higher pooled coverage but wider and less locally stable intervals,
while the PID variants did not provide a clear enough improvement to justify their
additional complexity. ACI provided the strongest overall balance of coverage, width,
tail balance, local stability and implementation simplicity.

This choice should not be interpreted as evidence that ACI is universally superior.
The 2023 through March 2025 period is used to compare the calibration methods, so ACI
is the preferred method for this analysis rather than the winner of a new untouched
test set.

## Interpretation of probabilistic forecasts

I interpret the probabilistic forecasts primarily at the prediction level rather than
through individual Linear QR coefficients.

Matrix B contains correlated demand lags, rolling statistics, calendar terms and weather
predictors. Because the quantile regressions are unregularized, individual coefficient
estimates are not the most stable basis for substantive interpretation.

Instead, I compare predicted medians, raw quantile spreads, calibrated interval widths
and empirical coverage across observable weather and demand regimes.

This analysis showed that warmer prior-day conditions are associated with both higher
predicted demand and wider forecast distributions. Average raw quantile-regression
interval width increased from approximately 1.66 MGD in the coolest temperature
quintile to approximately 4.04 MGD in the warmest.

Most of this weather-dependent variation was already present in the conditional
quantile forecasts. ACI generally supplied a smaller sequential correction based on
recent forecast errors rather than creating the weather-dependent uncertainty pattern
itself.

The widest forecast intervals also formed a recognizable high-uncertainty regime:
they were concentrated in warm, high-demand conditions and occurred predominantly
during summer. Wide intervals should therefore not automatically be interpreted as
forecast failures. In many cases, they indicate that the forecasting system has
recognized a more difficult operating regime.

These relationships are descriptive and predictive rather than causal. The analysis
does not estimate the effect that changing temperature, precipitation or other weather
conditions would have on water consumption.

## Known limitations

This project uses one municipal demand series and one primary weather station. A single
station cannot represent every local weather pattern across a utility service area.

The dataset ends on March 31, 2026, so 2026 is not a complete calendar year.

The weather inputs are historical observations. The operational feature set uses lagged
weather to preserve forecast-time availability, but a deployable system would ideally
incorporate archived one-day-ahead weather forecasts and their uncertainty.

The probabilistic analysis is based on a modest daily dataset and one municipal system.
Although sequential calibration substantially improves overall empirical coverage,
coverage is not uniform across every season, weather regime or demand level. Some
summer, wet-prior-day and intermediate-demand regimes remain more difficult.

The original point-model holdout has been opened and will not be reused as an untouched
test set. Later analyses are treated as separate extensions rather than retroactive
optimization of the reported holdout result.

Fort Collins' open-data platform was undergoing a migration when I acquired the
water-demand data on August 3, 2026. The project therefore preserves dated raw snapshots
and acquisition metadata so that future updates can be compared against the current
files.

These limitations are consistent with the intended scope of the project. The results
should be viewed as an analytical demonstration rather than an operational
water-demand planning tool.
