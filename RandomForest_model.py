from sklearn.ensemble import RandomForestRegressor
import pandas as pd
df = pd.read_csv("data\\feature_engineered_aapl.csv")
features = ['Open','High','Low','Volume','MA20','MA50','Daily_Return','Volatility','Lag_1','Lag_5','Lag_10','Price_Range']
target = 'Close'
from sklearn.model_selection import train_test_split
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

#Training RandomForest Model
rf = RandomForestRegressor(n_estimators=100,random_state=42)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
#Evaluate the model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("MAE:", mean_absolute_error(y_test, rf_pred))
print("R2:", r2_score(y_test, rf_pred))
mse = mean_squared_error(y_test, rf_pred)
print("MSE:", mse)