import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
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

#Training ARIMA model
train = df['Close'][:-30]
test = df['Close'][-30:]
model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()
predictions = model_fit.forecast(steps=30)
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(test, predictions)
print("MAE:", mae)

#Plotting
plt.figure(figsize=(12,6))

plt.plot(range(len(test)), test.values, label="Actual")
plt.plot(range(len(predictions)), predictions, label="Predicted")

plt.legend()
plt.title("ARIMA: Actual vs Predicted")
plt.xlabel("Days")
plt.ylabel("Close Price")

plt.show()

#Checking the metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
mae = mean_absolute_error(test, predictions)
mse = mean_squared_error(test, predictions)
r2 = r2_score(test, predictions)
print("MAE:", mae)
print("MSE:", mse)
print("R2:", r2)