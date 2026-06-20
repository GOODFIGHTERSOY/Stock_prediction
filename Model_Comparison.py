import pandas as pd

comparison = pd.DataFrame({"Model": ["Linear Regression","Random Forest","ARIMA"],"MAE": [0.61,18.22,18.85],"MSE": [0.61,810.04,443],
    "R2 Score": [0.99,-0.09,-4.03]})

print(comparison)