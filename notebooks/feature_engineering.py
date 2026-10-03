from pathlib import Path
import pandas as pd


# ============================================================
# PATHS
# ============================================================

RAW_FILE = Path("data/aapl.csv")
OUTPUT_FILE = Path("data/feature_engineered_aapl.csv")


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(RAW_FILE)

# Yahoo Finance CSV contains "Price" as the first column name
if "Price" in df.columns:
    df = df.rename(columns={"Price": "Date"})


# ============================================================
# CLEAN DATA
# ============================================================

# The CSV contains extra rows such as "Ticker" and "Date".
# Converting invalid values to NaT allows us to remove them
# without hard-coding df.iloc[2:] or df.iloc[3:].

df["Date"] = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df = df.dropna(subset=["Date"])

numeric_cols = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

df = (
    df
    .sort_values("Date")
    .drop_duplicates(subset=["Date"])
    .reset_index(drop=True)
)

# Handle missing numeric values
df[numeric_cols] = (
    df[numeric_cols]
    .interpolate(method="linear")
    .ffill()
    .bfill()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

# Moving averages
df["MA20"] = (
    df["Close"]
    .rolling(window=20)
    .mean()
)

df["MA50"] = (
    df["Close"]
    .rolling(window=50)
    .mean()
)

# Daily percentage return
df["Daily_Return"] = (
    df["Close"]
    .pct_change()
)

# 20-day rolling volatility
df["Volatility"] = (
    df["Daily_Return"]
    .rolling(window=20)
    .std()
)

# Intraday trading range
df["Price_Range"] = (
    df["High"] - df["Low"]
)

# Historical closing-price lags
df["Lag_1"] = df["Close"].shift(1)
df["Lag_5"] = df["Close"].shift(5)
df["Lag_10"] = df["Close"].shift(10)


# ============================================================
# NEXT-DAY TARGET
# ============================================================

# Today's features -> Tomorrow's Close
df["Target_Next_Close"] = (
    df["Close"]
    .shift(-1)
)

df["Target_Date"] = (
    df["Date"]
    .shift(-1)
)


# ============================================================
# REMOVE INCOMPLETE ROWS
# ============================================================

required_columns = [
    "MA20",
    "MA50",
    "Daily_Return",
    "Volatility",
    "Lag_1",
    "Lag_5",
    "Lag_10",
    "Target_Next_Close",
    "Target_Date"
]

df = (
    df
    .dropna(subset=required_columns)
    .reset_index(drop=True)
)


# ============================================================
# LEAKAGE SAFETY CHECK
# ============================================================

assert (
    df["Target_Date"] > df["Date"]
).all(), "Target date must always be after feature date."


# ============================================================
# SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("Feature engineering completed successfully.")
print("Rows:", len(df))
print("Date range:", df["Date"].min(), "to", df["Date"].max())
print("Saved:", OUTPUT_FILE)