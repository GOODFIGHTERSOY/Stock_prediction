# 📈 Stock Price Forecasting using Machine Learning & Time Series Analysis

> A Data Science project focused on forecasting Apple Inc. (AAPL) stock prices using Machine Learning and Time Series Analysis techniques.

---

## 🚀 Project Overview

The stock market is highly dynamic and influenced by numerous factors. This project aims to analyze historical stock data and develop predictive models capable of forecasting future stock prices.

The project follows a complete Data Science workflow:

- Data Collection
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Development
- Time Series Forecasting
- Model Evaluation & Comparison

This project was completed as part of my **Data Science Internship at Navoditha Infotech**.

---

## 🎯 Objectives

- Collect historical stock market data
- Analyze stock price trends and patterns
- Perform data preprocessing and feature engineering
- Build forecasting models
- Compare model performances
- Predict future stock prices

---

## 📂 Project Structure

```text
Stock_Prediction/
│
├── data/
│   ├── aapl.csv
|   ├──collection.py
│   └── featured_engineered_aapl.csv
│
├── notebooks/│   
│   ├── EDA.py
│   ├── feature_engineering.py
|   ├──Model_Training.py   
│   ├── Random_forest.py
│   ├── ARIMA_model.py
│   └── Model_comparison.py
└── README.md
```

---

## 📊 Dataset Information

### Source
Yahoo Finance API using the `yfinance` Python library.

### Dataset Features

| Feature | Description |
|----------|------------|
| Date | Trading Date |
| Open | Opening Stock Price |
| High | Highest Price of the Day |
| Low | Lowest Price of the Day |
| Close | Closing Stock Price |
| Volume | Number of Shares Traded |

### Time Period

2015 – 2025

---

## 🛠️ Technologies Used

### Programming Language

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- Statsmodels
- yFinance

---

## 🔍 Exploratory Data Analysis

The following analyses were performed:

- Closing Price Trend Analysis
- Trading Volume Analysis
- Moving Average Analysis
- Correlation Analysis
- Statistical Summary

### Key Observation

Apple stock demonstrated a strong long-term upward trend with periodic fluctuations influenced by market conditions.

---

## ⚙️ Feature Engineering

The following features were created to improve forecasting performance:

### Moving Averages

- MA20 (20-Day Moving Average)
- MA50 (50-Day Moving Average)

### Additional Features

- Daily Returns
- Lag Features
- Historical Trend Indicators

These engineered features helped models learn stock behavior more effectively.

---

## 🤖 Models Implemented

### 1. Linear Regression

A baseline machine learning model used for stock price prediction.

### 2. Random Forest Regressor

An ensemble-based machine learning model capable of capturing nonlinear patterns.

### 3. ARIMA

A statistical time-series forecasting model used to predict future stock prices.

---

## 📈 Model Performance

| Model | MAE | MSE | R² Score |
|---------|---------|---------|---------|
| Linear Regression | 0.61 | 0.61 | 0.99 |
| Random Forest | 18.22 | 810.04 | -0.09 |
| ARIMA | 18.85 | 443.00 | -4.03 |

---

## 🏆 Best Performing Model

### Linear Regression

Performance Metrics:

- MAE: 0.61
- MSE: 0.61
- R² Score: 0.99

The Linear Regression model achieved the highest predictive accuracy among all implemented models.

---

## 📷 Project Results

### Visualizations Generated

- Historical Stock Price Trend
- Trading Volume Analysis
- Moving Average Analysis
- Actual vs Predicted Comparison
- ARIMA Forecast Visualization

> Add screenshots of your generated graphs inside the repository and reference them here.

---

## 💡 Key Learnings

Through this project, I gained hands-on experience in:

- Data Cleaning and Preprocessing
- Time Series Analysis
- Feature Engineering
- Machine Learning Model Development
- Model Evaluation
- Financial Data Analytics

---

## 🔮 Future Scope

Potential enhancements include:

- LSTM-based Deep Learning Forecasting
- XGBoost Models
- Real-Time Stock Prediction
- News Sentiment Analysis
- Streamlit Dashboard Deployment
- Multi-Stock Forecasting System

---

## ⚡ Installation & Usage

### Clone Repository

```bash
git clone https://github.com/GOODFIGHTERSOY/Stock_prediction.git
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn statsmodels yfinance
```

### Run Data Collection

```bash
python notebooks/collection.py
```

### Run EDA

```bash
python notebooks/EDA.py
```

### Run Feature Engineering

```bash
python notebooks/feature_engineering.py
```

### Run Models

```bash
python notebooks/Model_Training.py

python notebooks/random_forest.py

python notebooks/ARIMA_model.py
```

---

## 👨‍💻 Author

**Krithik**



Interests:
- Data Science
- Machine Learning
- Artificial Intelligence
- Financial Analytics

---

## ⭐ Acknowledgements

- Yahoo Finance
- Scikit-Learn
- Statsmodels
- Python Community

---

### If you found this project interesting, consider giving it a ⭐ on GitHub!

📈 *Transforming historical stock data into actionable insights through Machine Learning and Time Series Forecasting.*
