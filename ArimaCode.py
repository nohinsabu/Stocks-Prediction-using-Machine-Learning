from sklearn import metrics
from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error
import pandas as pd

# Fit the model
df = pd.read_csv('TSLA.csv', index_col="Date", parse_dates=True)
model = ARIMA(df['Close'], order=(5, 1, 0))
model_fit = model.fit(disp=0)
residuals = DataFrame(model_fit.resid)

# For Visualing how the Time Series looks like
residuals.plot()
pyplot.show()
residuals.plot(kind='kde')
pyplot.show()
print(residuals.describe())

# Creating dataframe
dt = pd.DataFrame(index=range(0, len(df)), columns=['Date'])
dt = []
for i in range(0, len(df)):
    dt.append(df.index[i])

series = df
X = series['Close'].values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
dt_test = dt[size:len(X)]

history = [x for x in train]
predictions = list()
for t in range(len(test)):
    dt_prt = dt_test[t]
    model = ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs) + '     ' + str(dt_prt))

# Finding the error value
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)

#  For plotting the graph
pyplot.figure(figsize=(100, 8))
pyplot.plot(test, color='blue', label='testing data')
pyplot.plot(predictions, color='red', label='predicted data')
pyplot.legend()
pyplot.show()

np.sqrt(metrics.mean_squared_error(test, predictions))
