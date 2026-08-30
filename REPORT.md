# Car Price Prediction: Machine-Learning Report

**Student:** Lin Htet Aung  
**Task:** Predict the selling price of a used car from its vehicle and listing characteristics.

## 1. Dataset and objective

The original `Cars.csv` dataset contains 8,128 observations and 13 columns. The response variable is `selling_price`, measured in Indian rupees. Available predictors describe the car's brand, age, distance driven, fuel type, seller type, transmission, ownership history, mileage, engine capacity, maximum power, torque, and number of seats.

This is a supervised regression problem because the target is a continuous numeric value. The objective is to learn a function that maps the vehicle attributes to an estimated selling price and deploy the fitted model in a simple web application.

## 2. Data cleaning and feature preparation

The notebook applies the following cleaning steps:

1. CNG and LPG vehicles are removed because their mileage is recorded using a different unit from petrol and diesel vehicles.
2. Ownership categories are converted to an ordered numeric scale: first owner = 1 through fourth-or-above owner = 4. Test-drive cars are excluded because they are a small, atypically expensive group.
3. Numeric values are extracted from the text fields `mileage`, `engine`, and `max_power`; their units become km/l, CC, and bhp respectively.
4. The first word of the full car name is retained as the `brand` feature.
5. `torque` is removed because its many inconsistent textual formats would require substantial additional parsing.
6. Rows missing the required measurement fields are removed, as they represent only a small percentage of the data.
7. Exact duplicate rows are removed to avoid giving repeated listings excessive influence.

The model uses seven numeric predictors (`year`, `km_driven`, `owner`, `mileage`, `engine`, `max_power`, and `seats`) and four categorical predictors (`brand`, `fuel`, `seller_type`, and `transmission`).

The selling-price distribution is strongly right-skewed, so the natural logarithm of price is used as the training target. This reduces the influence of extremely expensive cars and makes proportional errors more meaningful. Predictions are converted back to rupees with the exponential function.

## 3. Exploratory data analysis

The exploratory plots show that newer cars and cars with greater maximum power generally have higher selling prices. More kilometres driven and a larger owner number generally correspond to lower prices. Engine size, transmission, fuel type, and brand also contain useful price information. The log transformation produces a more symmetric target distribution than raw selling price.

No predictor is discarded only because of low Pearson correlation. Correlation measures linear marginal relationships, whereas nonlinear models can learn interactions and curved relationships. Similarly, correlated features such as engine size and maximum power are retained because they describe different physical characteristics.

## 4. Train-test split and leakage prevention

The cleaned data are divided into an 80% training partition and a 20% held-out test partition using `random_state=42` for reproducibility. Model comparison and hyperparameter selection use only the training partition. The test partition is evaluated once after the final model is chosen.

Preprocessing is included inside a scikit-learn `Pipeline` and `ColumnTransformer`. Numeric fields use median imputation and standardization. Categorical fields use most-frequent imputation and one-hot encoding, with unknown categories ignored. Because every cross-validation fold fits its own pipeline, statistics and category encodings are learned only from that fold's training data, preventing data leakage.

## 5. Candidate models and cross-validation

Six regression approaches are compared with shuffled five-fold cross-validation:

- **Dummy median:** a baseline that always predicts the median training target.
- **Linear regression:** a simple additive linear relationship.
- **Support vector regression (RBF):** a nonlinear kernel method.
- **K-nearest neighbours:** predicts from similar training observations.
- **Decision tree:** learns nonlinear rules and interactions.
- **Random Forest:** averages many randomized trees to reduce variance and capture nonlinear relationships.

Models are compared using mean absolute error (MAE), root mean squared error (RMSE), and R² on the log-price scale. Lower MAE and RMSE are better; higher R² is better. Random Forest is the best untuned candidate, with mean cross-validated log-MAE **0.1638**, log-RMSE **0.2250**, and R² **0.9105**. The dummy baseline has log-RMSE **0.7550** and negative R², confirming that the trained models add substantial predictive value.

## 6. Hyperparameter tuning

Random Forest is tuned with `GridSearchCV`. The search considers 100 or 200 trees; maximum depths of 10, 20, or unlimited; and minimum leaf sizes of 1 or 2, with bootstrap sampling enabled. The best configuration is:

- `n_estimators = 200`
- `max_depth = 20`
- `min_samples_leaf = 2`
- `bootstrap = True`

Its best cross-validated log-RMSE is **0.2221**. A Random Forest is suitable because car pricing contains nonlinear effects and interactions—for example, the effect of model year can depend on brand, engine, and power.

## 7. Held-out test results

On the untouched test set, the final pipeline achieves:

| Metric | Result |
|---|---:|
| MAE (price scale) | ₹71,126 |
| RMSE (price scale) | ₹122,173 |
| R² (price scale) | 0.9297 |
| MAE (log scale) | 0.1552 |
| RMSE (log scale) | 0.2128 |
| R² (log scale) | 0.9168 |

The model explains about 93% of the variation in held-out selling prices. RMSE is larger than MAE because a small number of unusual or luxury cars have much larger errors. Results are strong for this dataset, but they should not be interpreted as guaranteed real-world valuation accuracy.

## 8. Feature importance and limitations

Both built-in tree importance and held-out permutation importance identify `year` and `max_power` as the strongest predictors, followed by `engine`. Brand, mileage, and kilometres driven also contribute. Low importance does not prove that a field has no value because correlated predictors may substitute for one another and rare categories have limited test support.

Important limitations include the random split, possible repeated listings that are not exact duplicates, sparse coverage of rare brands and luxury vehicles, and no measure of prediction uncertainty. Prices also change over time, so a deployed model can become outdated. Future work could use grouped or time-aware validation, more extensive tuning with nested cross-validation, richer vehicle-model features, and prediction intervals.

## 9. Deployment

The fitted preprocessing and Random Forest model are saved together as one `joblib` pipeline. The Streamlit app loads that artifact, converts form inputs into a one-row DataFrame, predicts log price, and transforms the result back to rupees. Packaging preprocessing with the estimator keeps inference consistent with training. Docker supplies a reproducible runtime and exposes the Streamlit service on port 8501.
