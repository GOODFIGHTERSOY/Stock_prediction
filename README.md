# 📈 Apple Stock Price Forecasting

A Data Science project focused on predicting the **next trading day's closing price of Apple (AAPL)** using Machine Learning and Time-Series models.

The project compares:

- Linear Regression
- Random Forest Regressor
- ARIMA
- Persistence Baseline

---

## 🎯 Objective

The goal of this project is to use historical stock-market information available on Day `t` to predict the closing price on Day `t+1`.

```text
Historical AAPL Data
        ↓
Data Cleaning
        ↓
Feature Engineering
        ↓
Chronological Train/Test Split
        ↓
Machine Learning + Time-Series Models
        ↓
Next-Day Price Prediction
        ↓
Model Evaluation
