# Model Card: Used Car Price Estimator

## Model purpose

Predict the selling price of a used-car listing from raw marketplace-style inputs.

## Inputs

- `name`
- `year`
- `km_driven`
- `fuel`
- `seller_type`
- `transmission`
- `owner`
- `mileage`
- `engine`
- `max_power`
- `torque`
- `seats`

## Training pipeline

- removed exact duplicate rows before splitting
- used a train / validation / test workflow
- engineered parsed spec features from semi-structured text columns
- compared multiple regressors including linear, regularised, tree, and boosting models
- selected the champion model on validation performance and evaluated once on the untouched test split

## Champion model

- `HistGradientBoostingRegressor` inside a preprocessing and target-transformation pipeline

## Final test metrics

- `R2`: `0.897`
- `MAE`: `73,453`
- `RMSE`: `164,725`

## Important signals

Top raw input drivers in the notebook analysis included:

- `max_power`
- `year`
- `torque`
- `name`
- `transmission`
- `km_driven`

## Known limitations

- automatic cars show larger absolute errors because they tend to sit in higher price bands
- rare fuel types such as LPG and CNG have small sample sizes
- the model is strongest when the input format resembles a realistic listing
- this is a pricing support tool, not a formal valuation guarantee
