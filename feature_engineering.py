import pandas as pd
#Features
df = pd.read_csv("data/aapl.csv")
df = df.iloc[3:].copy()
df.rename(columns={'Price': 'Date'}, inplace=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].rolling(20).std()
df['Price_Range'] = df['High'] - df['Low']
df['Lag_1'] = df['Close'].shift(1)
df['Lag_5'] = df['Close'].shift(5)
df['Lag_10'] = df['Close'].shift(10)

#Drop rows with NaN values
df.dropna(inplace=True)

#Save the engineered features to a new CSV file
df.to_csv("data/feature_engineered_aapl.csv", index=False)
