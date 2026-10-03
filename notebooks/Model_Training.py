from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# PATHS
# ============================================================

DATA_FILE = Path(
    "data/feature_engineered_aapl.csv"
)

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(DATA_FILE)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Target_Date"] = pd.to_datetime(
    df["Target_Date"]
)


# ============================================================
# FEATURES AND TARGET
# ============================================================

features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "MA20",
    "MA50",
    "Daily_Return",
    "Volatility",
    "Lag_1",
    "Lag_5",
    "Lag_10",
    "Price_Range"
]

target = "Target_Next_Close"


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)

train = df.iloc[
    :split_index
].copy()

test = df.iloc[
    split_index:
].copy()

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


print(
    "Training samples:",
    len(train)
)

print(
    "Testing samples:",
    len(test)
)

print(
    "Train period:",
    train["Date"].min(),
    "to",
    train["Date"].max()
)

print(
    "Test period:",
    test["Date"].min(),
    "to",
    test["Date"].max()
)


# ============================================================
# METRIC FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    predictions,
    current_close
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    mse = mean_squared_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_true,
        predictions
    )

    # Did the model correctly predict
    # whether tomorrow would close above/below today?
    actual_direction = (
        y_true.to_numpy()
        >
        current_close.to_numpy()
    )

    predicted_direction = (
        np.asarray(predictions)
        >
        current_close.to_numpy()
    )

    directional_accuracy = (
        actual_direction
        ==
        predicted_direction
    ).mean() * 100

    return {
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy": directional_accuracy
    }


# ============================================================
# BASELINE MODEL
# ============================================================

# Simple persistence baseline:
# Tomorrow's Close = Today's Close

baseline_predictions = (
    X_test["Close"]
    .to_numpy()
)

baseline_result = evaluate_model(
    "Persistence Baseline",
    y_test,
    baseline_predictions,
    X_test["Close"]
)


# ============================================================
# LINEAR REGRESSION
# ============================================================

model = LinearRegression()

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

linear_result = evaluate_model(
    "Linear Regression",
    y_test,
    predictions,
    X_test["Close"]
)


# ============================================================
# RESULTS
# ============================================================

results = pd.DataFrame(
    [
        baseline_result,
        linear_result
    ]
)

print()
print("=" * 70)
print("MODEL RESULTS")
print("=" * 70)
print(results)


results.to_csv(
    OUTPUT_DIR / "linear_regression_metrics.csv",
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "Feature_Date": test["Date"].values,
    "Target_Date": test["Target_Date"].values,
    "Current_Close": test["Close"].values,
    "Actual_Next_Close": y_test.values,
    "Predicted_Next_Close": predictions,
    "Baseline_Prediction": baseline_predictions
})

prediction_df.to_csv(
    OUTPUT_DIR /
    "linear_regression_predictions.csv",
    index=False
)


# ============================================================
# PLOT
# ============================================================

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test["Target_Date"],
    y_test,
    label="Actual Next-Day Close"
)

plt.plot(
    test["Target_Date"],
    predictions,
    label="Linear Regression"
)

plt.plot(
    test["Target_Date"],
    baseline_predictions,
    label="Persistence Baseline",
    alpha=0.6
)

plt.xlabel("Date")
plt.ylabel("AAPL Close Price")
plt.title(
    "Next-Day AAPL Forecast: "
    "Actual vs Predicted"
)

plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR /
    "linear_regression_forecast.png",
    dpi=200
)

plt.show()