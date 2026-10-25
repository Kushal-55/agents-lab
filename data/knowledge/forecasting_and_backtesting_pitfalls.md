# Market forecasting and backtesting pitfalls

## Markets are close to efficient
Prices already reflect most public information, so predicting the next day's or week's direction is very hard. The chance that a liquid asset closes higher after five trading days is mostly the historical share of up-moves, with only a small, unstable extra signal on top. A model that claims 70 percent accuracy on daily direction should be assumed to contain a bug until proven otherwise.

## Look-ahead bias
Look-ahead bias means using information that was not available at the time of the decision, for example normalising with statistics from the whole dataset or training on labels that are not yet known. It makes backtests look excellent and live results poor. The fix is walk-forward testing, where each prediction uses only data available on that date.

## Overfitting and data snooping
Trying many strategies and keeping the best one guarantees a good-looking backtest by luck alone. Use out-of-sample data, keep models simple, and count how many ideas were tried.

## Baselines
Always compare a model with naive baselines: always predict the more common outcome, and predict the historical base rate probability. A forecast is useful only if it beats these on out-of-sample data.

## Probabilistic scoring
The Brier score is the mean squared difference between a forecast probability and the 0 or 1 outcome. Lower is better. The Brier skill score compares it with a baseline forecast; a value above zero means the model beats the baseline.

## Overlapping samples
When predicting five days ahead from daily data, consecutive samples overlap and are not independent. The effective sample size is roughly the number of samples divided by the horizon, so statistical significance is weaker than it looks.

## Transaction costs and regime change
Real trading pays spreads, fees and taxes, and relationships that worked in one market regime can vanish in the next. A small statistical edge can disappear after costs.
