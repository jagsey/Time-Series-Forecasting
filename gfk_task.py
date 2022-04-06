import seaborn as sns
import pandas as pd
import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import mean_squared_error
from scipy import stats
from pmdarima import model_selection

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

import os; os.chdir('/home/jagsy/PycharmProjects/insta/gfk/Neo team hiring home task')

# Convert the data from 10 minute to hourly interval

data = pd.read_csv("taxi.csv")
data.datetime = pd.to_datetime(data.datetime, dayfirst=True)
data.set_index('datetime',inplace=True)
data = data.resample('H').mean()


plt.plot(data)
plt.title('Original Data')
plt.ylabel('Hourly Bookings')
plt.show()

#Box Plot to check the outliers
sns.boxplot(data.num_orders, )
plt.title('Box Plot to check Outliers')
plt.show()

#IQR to find and remove outliers
print(data.describe())
Q1 = np.percentile(data['num_orders'], 25,
                   interpolation='midpoint')

Q3 = np.percentile(data['num_orders'], 75,
                   interpolation='midpoint')
IQR = Q3 - Q1
print(IQR)

# Above Upper bound
upper = data['num_orders'] >= (Q3 + 3.5 * IQR)

print("Upper bound:", upper)
print(np.where(upper)[0])

print("New Shape: ", data.shape)

data.iloc[list(np.where(upper)[0])] = data.mean()

plt.plot(data)
plt.title('Outliers removed')
plt.show()


data = data[len(data) - 480:]
data_train, data_test = model_selection.train_test_split(data, train_size=432)

data_train[data_train.num_orders==0]=1
data_test[data_test.num_orders==0]=1

def box_cox(data):

    # Box Cox transformer and Data Standardization
    pt = PowerTransformer(method='box-cox')
    pt.fit(data)
    print(pt.lambdas_)
    transform_power = pt.transform(data)
    scaler = StandardScaler()
    scaler = scaler.fit(transform_power)
    print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, np.sqrt(scaler.var_)))

    normalized = scaler.transform(transform_power)
    transform_power_df = pd.DataFrame({'num_orders':normalized.flatten()}, index=data.index)

    # plt.plot(transform_power_df, )
    # plt.title('Scaled and Transformed')
    # plt.show()

    return transform_power_df

box_cox_train = box_cox(data_train)
box_cox_test = box_cox(data_test)


# data transformation
data_train_differenced = box_cox_train.diff().dropna()
plt.plot(data_train_differenced)
plt.title('Differenced data')
plt.show()

transformed_differenced_seasonal = box_cox_train.diff().diff(24).dropna()
plt.plot(transformed_differenced_seasonal)
plt.title('Seasonally differenced data')
plt.show()

# Check the Seasonality of the Data with Augmented Dickey-Fuller Test. If the p-value of the test is less
# than 5%, the time series is stationary.
adk = adfuller(data_train_differenced)
print(adk)

# Auto correlation is non zero at different lags. Therefore, the series can
# be forecast from the past. The plot also displays seasonal autocorrelation at lags 24.
# Seasonal Arima can be used to model the time series with seasonal value of 24..

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8,8))
plot_acf(data_train_differenced, lags=50, zero=False, ax=ax1)
plt.title('Autocorrelation')
plot_pacf(data_train_differenced, lags=50, zero=False, ax=ax2)
plt.title('Partial Autocorrelation')
plt.show()


fig, (axx1, axx2) = plt.subplots(2, 1, figsize=(8,8))
plot_acf(transformed_differenced_seasonal, lags=50, zero=False, ax=axx1)
plt.title('Seasonal Autocorrelation')
plot_pacf(transformed_differenced_seasonal, lags=50, zero=False, ax=axx2)
plt.title('Seasonal Partial Autocorrelation')
plt.show()

# Decomposition of the time series for observing different components such as Trend, Seasonality and Residual
decomposition = sm.tsa.seasonal_decompose(box_cox_train)

# Decomposing all the components and visualize them
decomposition.plot()

model = SARIMAX(data_train, order=(1,0,1), seasonal_order=(0,1,1,24),)
results = model.fit(maxiter=50)
print(results.summary())
print('Mean Residual of SARIMA is: ', results.resid.mean())
results.plot_diagnostics()
plt.show()

future_forecast = results.get_prediction(start='2018-08-13 15:00:00', end='2018-08-31 23:00:00', )
predictions = future_forecast.predicted_mean
print(predictions, data_test.num_orders,)
rmse = np.sqrt((predictions - data_test.num_orders) ** 2).mean()
print("\\nRMSE of ARIMA model is: ",rmse)

forecast = results.get_forecast(steps=48)
ci = forecast.conf_int()

figs = data_train.plot(label = 'Original')
figs.set_xlabel('Date')
figs.set_ylabel('Sales')
figs.fill_between(ci.index, ci.iloc[:,0], ci.iloc[:, 1], color='k', alpha=.2)

predictions.plot(ax=figs, label='Predictions', alpha=.7)

plt.legend()
plt.show()

# Holtz Winter Exponential Smoothing for dealing with Seasonality and Trend

hwmodel = ExponentialSmoothing(data_train, trend='add', seasonal='mul',
                               seasonal_periods=24).fit()

predictions_hw = hwmodel.forecast(48)

data_test.plot(legend=True, label = 'Test', color='blue')
predictions_hw.plot( label='Predictions HW', alpha=.7, color='green')

plt.legend()
plt.show()


print("RMSE of the Exponential Average Model is: ", np.sqrt(mean_squared_error(data_test, predictions_hw)))

import itertools
p = d = q = range(0,2)

pdq = list(itertools.product(p,d,q))
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p,d,q))]

# Script to determine the order of Seasonal ARIMA model with minimal AIC value

# order_aic_bic = []
# for pm in pdq:
#     for pm_seasonal in seasonal_pdq:
#         model = SARIMAX(data_train, order=pm, seasonal_order=pm_seasonal)
#         results = model.fit()
#         order_aic_bic.append((pm, pm_seasonal, results.aic, results.bic))
#         print('\n\n')
#         print(pm, pm_seasonal)
# order_df = pd.DataFrame(order_aic_bic, columns=['p','q','aic','bic'])
# print(order_df.sort_values('aic'))
