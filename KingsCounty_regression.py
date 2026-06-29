import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import r2_score, mean_squared_error

"""
1. Linear Regression(Ordinary)
-Gives a baseline perfomance.
-Uses unbiased coefficients.
-Has high variance when features correlate.
-Tends to overfitt
-Coefficient instability

2.Ridge Regression
-Penalizes large coefficients.
-Keep all features, but tames them.
-Excellent when predictors move together
-Coefficients shrink but do not dissapear.
-Beter generalization than Ordinary Linear Regression.

3.Lasso Regression
-Perfoms automatic feature selection.
-Pushes weak predictors to exactly zero.
-Reveals what actually matters.

*CV(cross-validation)
-It is a way to test how well a model will perform on unseen data, using on;y the training set.
-Instead of training once and hoping the model generalizes.
-It splits the training data into several parts (called folds).
-Train on some parts.
-Validate on the remaining part.
-Repeat this multiple times
-Average the results

*Alpha
-Alpha (α) controls how strongly the model is penalized for large coefficients.
-Alpha decides how much the model is allowed to “care” about each feature.

*Alpha = 0
-No penalty
-Same as ordinary least squares
-High variance, overfitting risk

*Small alpha (e.g. 0.001)
-Very light penalty
-Model keeps most features
-Coefficients stay large

*Medium alpha (sweet spot)
-Coefficients shrink
-Noise features get reduced
-Best generalization

*Large alpha (e.g. 10, 100)
-Strong penalty
-Model becomes simple
-Lasso may zero out many features
-Risk of underfitting
"""

df = pd.read_csv("house_data.csv")

features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "grade",
    "sqft_above", "sqft_basement", "lat", "long",
    "yr_built"
]

X = df[features]
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

ols = LinearRegression()

ridge_cv = RidgeCV(
    alphas=[0.1, 1.0, 10.0],
    cv=5
)

lasso_cv = LassoCV(
    alphas=[0.001, 0.01, 0.1],
    cv=5,
    max_iter=10000
)

ols.fit(X_train_scaled, y_train)
ridge_cv.fit(X_train_scaled, y_train)
lasso_cv.fit(X_train_scaled, y_train)

models = {
    "OLS": ols,
    "Ridge (CV)": ridge_cv,
    "Lasso (CV)": lasso_cv
}

for name, model in models.items():
    preds = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"{name}")
    print(f"root_mean_square_error = {rmse:.0f}")
    print(f"R2={r2:.3f}\n")

print("\nChosen regularization strengths:")
print(f"Ridge alpha: {ridge_cv.alpha_}")
print(f"Lasso alpha: {lasso_cv.alpha_}")

"""
Key Takeaways:
    *Predictive Power
    -All models explain approximately 69.6% of the variance in house prices. An RMSE of ~$217k suggests the "average" error in your price predictions is quite high, which is typical for King County given the high-value outliers seen in your geospatial heatmap.

    *Regularization Impact
    -The chosen Lasso alpha (0.1) is very low, and the Ridge alpha (10.0) is also relatively small. This indicates that heavy regularization wasn't necessary to prevent overfitting, but it also means these models aren't significantly different from standard OLS.

    *The Tie
    -The fact that all models performed nearly the same suggests that the relationship between your features and house price is primarily linear, but there is a "ceiling" to how much a linear model can capture (likely due to the complex spatial factors seen in the Seattle heatmap).
"""