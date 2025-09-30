# Ex.No:04 FIT ARMA MODEL FOR TIME SERIES
# Date: 30.09.2025

### AIM:
To implement ARMA model in python.

### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.

### PROGRAM:
```python
###Name: HIRUTHIK SUDHAKAR
###Reg No: 212223240054

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
df = pd.read_csv("/mnt/data/BMW_Car_Sales_Classification.csv")

# Aggregate yearly sales
yearly_sales = df.groupby("Year")["Sales_Volume"].sum().reset_index()
X = yearly_sales["Sales_Volume"]

# Plot original data
plt.figure(figsize=(12,6))
plt.plot(yearly_sales["Year"], X, marker="o")
plt.title("BMW Yearly Sales Volume (2010–2024)")
plt.show()

# Check stationarity (ADF Test)
result = adfuller(X)
print("ADF Statistic:", result[0])
print("p-value:", result[1])

# Plot ACF and PACF
plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plot_acf(X, lags=5, ax=plt.gca())
plt.title("Original Data ACF")
plt.subplot(2,1,2)
plot_pacf(X, lags=5, ax=plt.gca())
plt.title("Original Data PACF")
plt.tight_layout()
plt.show()


# Fit ARMA(1,1) Model
arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
print("ARMA(1,1) Summary:\n", arma11_model.summary())

# Simulate ARMA(1,1) process
phi1 = arma11_model.params['ar.L1']
theta1 = arma11_model.params['ma.L1']
ar1 = np.array([1, -phi1])
ma1 = np.array([1, theta1])
arma11_sim = ArmaProcess(ar1, ma1).generate_sample(nsample=200)

plt.plot(arma11_sim)
plt.title("Simulated ARMA(1,1) Process")
plt.show()
plot_acf(arma11_sim)
plt.show()
plot_pacf(arma11_sim)
plt.show()

# Fit ARMA(2,2) Model
arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
print("ARMA(2,2) Summary:\n", arma22_model.summary())

# Simulate ARMA(2,2) process
phi1 = arma22_model.params['ar.L1']
phi2 = arma22_model.params['ar.L2']
theta1 = arma22_model.params['ma.L1']
theta2 = arma22_model.params['ma.L2']
ar2 = np.array([1, -phi1, -phi2])
ma2 = np.array([1, theta1, theta2])
arma22_sim = ArmaProcess(ar2, ma2).generate_sample(nsample=200)

plt.plot(arma22_sim)
plt.title("Simulated ARMA(2,2) Process")
plt.show()
plot_acf(arma22_sim)
plt.show()
plot_pacf(arma22_sim)
plt.show()

```

### OUTPUT:

### SIMULATED ARMA(1,1) PROCESS:

<img width="960" height="504" alt="image" src="https://github.com/user-attachments/assets/37d20b8f-09c7-4e4a-bfdb-e5b8733612a5" />
<br>
<br>
<img width="632" height="463" alt="image" src="https://github.com/user-attachments/assets/debbe117-53bc-4454-bc9d-660b4506a284" />
<br>
<br>

### Partial Autocorrelation
<br>
<br>
<img width="711" height="480" alt="image" src="https://github.com/user-attachments/assets/5dd773b8-0020-41fa-bce7-1ac2376f2ebf" />
<br>
<br>

### Autocorrelation
<br>
<br>
<img width="703" height="466" alt="image" src="https://github.com/user-attachments/assets/6f5d5dd6-c278-4043-bc79-8805b6ac3ade" />
<br>
<br>

### SIMULATED ARMA(2,2) PROCESS:
<br>
<br>
<img width="952" height="545" alt="image" src="https://github.com/user-attachments/assets/57aa35b7-2d96-4322-9075-05ba7810c752" />
<br>
<br>
<img width="603" height="466" alt="image" src="https://github.com/user-attachments/assets/2463abc3-7223-4265-8906-92e40ddd59d3" />
<br>
<br>

### Partial Autocorrelation
<br>
<br>
<img width="725" height="466" alt="image" src="https://github.com/user-attachments/assets/cb21b19f-04a4-4b47-ab03-181b6dfbc5c3" />
<br>
<br>


### Autocorrelation
<br>
<br>
<img width="668" height="473" alt="image" src="https://github.com/user-attachments/assets/91967679-f0bf-4cc8-9d46-209d01aa1edf" />
<br>



### RESULT:
Thus, a python program is created to fir ARMA Model successfully.
