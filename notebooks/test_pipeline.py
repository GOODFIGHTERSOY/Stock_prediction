import pandas as pd


df = pd.read_csv(
    "data/feature_engineered_aapl.csv"
)

df["Date"] = pd.to_datetime(
    df["Date"]
)

df["Target_Date"] = pd.to_datetime(
    df["Target_Date"]
)


# Target must always be in the future
assert (
    df["Target_Date"] > df["Date"]
).all()


required = [
    "Close",
    "MA20",
    "MA50",
    "Daily_Return",
    "Volatility",
    "Lag_1",
    "Lag_5",
    "Lag_10",
    "Target_Next_Close"
]


assert not (
    df[required]
    .isna()
    .any()
    .any()
)


print(
    "All pipeline checks passed."
)