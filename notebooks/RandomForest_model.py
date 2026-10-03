from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


DATA_FILE = Path(
    "data/feature_engineered_aapl.csv"
)

OUTPUT_DIR = Path("output")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


df = pd.read_csv(
    DATA_FILE
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Target_Date"] = pd.to_datetime(
    df["Target_Date"]
)


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
# SAME CHRONOLOGICAL SPLIT
# ============================================================

split_index = int(
    len(df) * 0.80
)

train = df.iloc[
    :split_index
]

test = df.iloc[
    split_index:
]

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    y_test,
    predictions
)


actual_direction = (
    y_test.to_numpy()
    >
    X_test["Close"].to_numpy()
)

predicted_direction = (
    predictions
    >
    X_test["Close"].to_numpy()
)

directional_accuracy = (
    actual_direction
    ==
    predicted_direction
).mean() * 100


results = pd.DataFrame([
    {
        "Model": "Random Forest",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "Directional_Accuracy":
            directional_accuracy
    }
])


print(results)


results.to_csv(
    OUTPUT_DIR /
    "random_forest_metrics.csv",
    index=False
)


prediction_df = pd.DataFrame({
    "Feature_Date": test["Date"].values,
    "Target_Date": test["Target_Date"].values,
    "Current_Close": test["Close"].values,
    "Actual_Next_Close": y_test.values,
    "Predicted_Next_Close": predictions
})

prediction_df.to_csv(
    OUTPUT_DIR /
    "random_forest_predictions.csv",
    index=False
)