import pandas as pd
df = pd.read_csv("data\\aapl.csv")
print(df.head())

# Rename first column to Date
df.rename(columns={'Price':'Date'}, inplace=True)

# Removing the first two junk rows
df = df.iloc[2:]

# Convert date
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

#Describing the data
print(df.describe())

#Checking for null values
print("Null values in each column:")
print(df.isnull().sum())
print(df.dtypes)
print(df.columns)
numeric_cols = ['Close', 'High', 'Low', 'Open', 'Volume']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

#Handling Null Values
Nv=["Close","High","Low","Open","Volume"]
df[Nv]= df[Nv].interpolate(method='linear', inplace=True)
df = df.interpolate(method='linear')
df = df.ffill()
df = df.bfill()
print("Null values after interpolation:")
print(df.isnull().sum())
print(df.head())

#EDA
print(df.shape)
print(df.info())
print(df.describe())
print(df.head())
print(df.columns)
df.set_index('Date', inplace=True)

#Closing price trend
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.plot(df.index, df["Close"])
plt.title('Apple Closing Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

#Trading volume analysis
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Volume'])
plt.title("Apple Trading Volume")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.show()

#Moving average
df['MA20'] = df['Close'].rolling(20).mean()
df['MA50'] = df['Close'].rolling(50).mean()
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Close'], label='Close Price')
plt.plot(df.index, df['MA20'], label='20-Day MA')
plt.plot(df.index, df['MA50'], label='50-Day MA')
plt.legend()
plt.title("Moving Average Analysis")
plt.show()
#Daily return
df['Daily_Return'] = df['Close'].pct_change() * 100
plt.figure(figsize=(14,6))
plt.plot(df.index, df['Daily_Return'])
plt.title("Daily Percentage Returns")
plt.show()