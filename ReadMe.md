# **`Forecasting of Hourly Bookings`**

_Analysis and Interpretations_
# Original Data

![Original_Data](https://user-images.githubusercontent.com/103120317/162107984-19f150a3-3a14-4ca8-b055-f56c0a34978f.png)

a) Data points are seperated by an interval of 10 mins.
b) The data points are aggregated inorder to convert them into hourly bookings.

# Visualize and Remove Outliers

![Box _plot](https://user-images.githubusercontent.com/103120317/162108339-c49b5f4c-98b1-4ed3-bce3-5538c473677e.png)

![Data_with_Outliers_removed](https://user-images.githubusercontent.com/103120317/162108378-a13f0cd0-b76f-408e-8db4-bed07eb7fffc.png)

a) The data is plotted to visualize outliers.
b) IQR is used to remove the outliers based on the first and second quartiles.

# Time Series Decomposition

![Seasonal_Decomposition](https://user-images.githubusercontent.com/103120317/162108521-94dbe5fb-c1fb-4ca5-b449-60386cb53d46.png)

a) Extraction of Seasonal patterns, Trend and Residuals.

# De-Trended Series

![de_Trended](https://user-images.githubusercontent.com/103120317/162108821-eb281f93-40fe-4c11-902f-16b22fe12855.png)

a) Data was transformed using a rolling average of 24 data-points.
b) Observed seasonal patterns at lags 24 and 48 in the ACF plot.

# 1. Transformation with Differencing
# 2. ACF and PACF plot

![Differenced_Data](https://user-images.githubusercontent.com/103120317/162108434-66fb1f3a-a034-4106-9deb-4256e2b9ba49.png)

![Acf_and_Pacf](https://user-images.githubusercontent.com/103120317/162109049-065c33e1-29b0-49fb-816a-7a55f56ce4b8.png)

a) Taken the first difference as data transformation to bring stationarity to the data.
b) The ACF and PACF plots were used to determine the model order.
c) The first differencing as well as ACF and PACF plots helps us determine the non-seasonal patterns.

# 1. Transformation with Seasonal Differencing
# 2. ACF and PACF plot of Seasonally differenced data

![Seasonally_Differenced_Data](https://user-images.githubusercontent.com/103120317/162109334-65894c39-a401-4429-9b2e-5fed29fe9a31.png)

![Seasonal_ACF_and _PACF](https://user-images.githubusercontent.com/103120317/162109402-48a4852b-1793-419b-add8-9e8624bfdc2b.png)

a) Taken the seasonal difference to the differenced data with a seasonal period of 24.
b) Visualizing ACF and PACF plots helps us determine the seasonal patterns of seasonally differenced data.

# SARIMA Modelling

![sarima_511](https://user-images.githubusercontent.com/103120317/162109442-b193f42a-8ce8-46c5-8e5b-fedda7645117.png)

a) Can capture both non-seasonal and seasonal patterns.
b) It is difficult to determine model order through ACF and PACF when both P and Q values are non zero.
c) AIC values - Good at choosing predictive models.
    -- Lower AIC score -> Better Model
    -- Helps to overcome overfitting.
d) Box-Cox transformation was applied to stabilize the variance.
e) The mean RMSE obtained was 0.5.

# SARIMA Model diagnostics

![sarima model diag 511](https://user-images.githubusercontent.com/103120317/162109678-2fb603c0-a345-46a7-b848-beac3e2c47c8.png)

a) Model Order – Non Seasonal (1,0,1); Seasonal (5,1,1,24).
b) Mean Residual – 0.05.
c) Residual looks like White Noise.
d) Prob(JB) - 0.16 -> Residual is close to Normal Distribution.
e) Q – Q plot is close to linear.
f) ACF in Correlogram shows no correlations at lag > 0. 
g) Prob(Q) - 0.79 -> Residuals are uncorrelated.

# Exponential Smoothing

![Forecasting_with_Holtz_Winter_ Method](https://user-images.githubusercontent.com/103120317/162110017-cd964fa3-1430-434a-b159-65b03be38976.png)

a) Model chosen as a result of Trend and Seasonal Characteristics.
b) Trend & Seasonal are - Additive.
c) Mean RMSE - 0.7
d) Seasonal ARIMA model performs better based on RMSE.

# Next Steps

a) Deep Learning approach – LSTMs. Can use CNN for feature extractor.
b) Data such as Booking Price and Holidays for Demand Forecasting.  Can use weather information as an influencial factor.
   -- Advanced models such as Vector Auto-Regression and Dynamic Harmonic Regression could be used.


