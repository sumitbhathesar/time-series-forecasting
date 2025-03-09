import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
values = np.cumsum(np.random.randn(100))  
df = pd.DataFrame({"Date": dates, "Value": values})
df.set_index("Date", inplace=True)

plt.figure(figsize=(10,5))
plt.plot(df.index, df["Value"], label="Actual Data")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Data")
plt.legend()
plt.show()
df["Lag_1"] = df["Value"].shift(1)  
df.dropna(inplace=True) 

X = df[["Lag_1"]].values  
Y = df["Value"].values  

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, shuffle=False)

model = LinearRegression()
model.fit(X_train, Y_train)

predictions = model.predict(X_test)

mae = mean_absolute_error(Y_test, predictions)
print(f"Mean Absolute Error: {mae:.4f}")

plt.figure(figsize=(10,5))
plt.plot(df.index[-len(Y_test):], Y_test, label="Actual", color='blue')
plt.plot(df.index[-len(Y_test):], predictions, label="Predicted", color='red')
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Time Series Forecasting (Linear Regression)")
plt.legend()
plt.show()
