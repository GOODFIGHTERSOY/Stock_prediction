import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ---------------------------------------------------------
# 1. CREATE OUTPUT FOLDER
# ---------------------------------------------------------

os.makedirs("output", exist_ok=True)


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv("data/aapl.csv")

df.rename(
    columns={"Price": "Date"},
    inplace=True
)


# ---------------------------------------------------------
# 3. CLEAN DATA
# ---------------------------------------------------------

# Your dates are like 02-01-2015
df["Date"] = pd.to_datetime(
    df["Date"],
    format="%d-%m-%Y",
    errors="coerce"
)

df["Close"] = pd.to_numeric(
    df["Close"],
    errors="coerce"
)

# Remove junk rows
df = df.dropna(
    subset=["Date", "Close"]
)

df = df.sort_values("Date")

df = df.reset_index(drop=True)


# ---------------------------------------------------------
# 4. TRAIN / TEST SPLIT
# ---------------------------------------------------------

split = int(len(df) * 0.80)

train = df.iloc[:split]
test = df.iloc[split:]

train_data = train["Close"].astype(float)
test_data = test["Close"].astype(float)


print("Training records:", len(train_data))
print("Testing records:", len(test_data))


# ---------------------------------------------------------
# 5. TRAIN ARIMA
# ---------------------------------------------------------

order = (0, 1, 1)

model = ARIMA(
    train_data,
    order=order
)

model_fit = model.fit()

print("ARIMA trained successfully")
print("Order:", order)


# ---------------------------------------------------------
# 6. WALK-FORWARD PREDICTION
# ---------------------------------------------------------

predictions = []

correct_direction = 0

for actual_price in test_data:

    # Previous known closing price
    previous_close = model_fit.data.endog[-1]

    # Predict only next day
    forecast = model_fit.forecast(
        steps=1
    )

    predicted_price = float(
        forecast.iloc[0]
    )

    predictions.append(
        predicted_price
    )

    # Direction check
    actual_direction = (
        actual_price > previous_close
    )

    predicted_direction = (
        predicted_price > previous_close
    )

    if actual_direction == predicted_direction:
        correct_direction += 1

    # Add actual value before next prediction
    model_fit = model_fit.append(
        [actual_price],
        refit=False
    )


predictions = np.array(
    predictions
)


# ---------------------------------------------------------
# 7. EVALUATION
# ---------------------------------------------------------

mae = mean_absolute_error(
    test_data,
    predictions
)

mse = mean_squared_error(
    test_data,
    predictions
)

rmse = np.sqrt(mse)

r2 = r2_score(
    test_data,
    predictions
)

direction_accuracy = (
    correct_direction
    / len(test_data)
) * 100


print("\nARIMA RESULTS")
print("----------------------------")

print("MAE :", round(mae, 2))
print("MSE :", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2  :", round(r2, 4))
print(
    "Directional Accuracy:",
    round(direction_accuracy, 2),
    "%"
)


# ---------------------------------------------------------
# 8. SAVE METRICS
# ---------------------------------------------------------

metrics = pd.DataFrame({
    "Model": ["ARIMA(0,1,1)"],
    "MAE": [mae],
    "MSE": [mse],
    "RMSE": [rmse],
    "R2": [r2],
    "Directional_Accuracy": [
        direction_accuracy
    ]
})

metrics.to_csv(
    "output/arima_metrics.csv",
    index=False
)


# ---------------------------------------------------------
# 9. SAVE PREDICTIONS
# ---------------------------------------------------------

results = pd.DataFrame({
    "Date": test["Date"].values,
    "Actual": test_data.values,
    "Predicted": predictions
})

results.to_csv(
    "output/arima_predictions.csv",
    index=False
)


# ---------------------------------------------------------
# 10. PLOT
# ---------------------------------------------------------

plt.figure(
    figsize=(14, 6)
)

plt.plot(
    test["Date"],
    test_data.values,
    label="Actual"
)

plt.plot(
    test["Date"],
    predictions,
    label="ARIMA Prediction"
)

plt.title(
    "ARIMA Walk-Forward Next-Day Forecast"
)

plt.xlabel("Date")
plt.ylabel("AAPL Close Price")

plt.legend()
plt.tight_layout()

plt.show()