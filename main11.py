
import random
import math
import os
import warnings
import nltk

from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import re
import preprocessor as p

import yfinance as yf
import datetime as dt
from datetime import datetime
from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
plt.style.use('ggplot')
nltk.download('punkt')

# Ignore Warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ***************** FLASK *****************************
app = Flask(__name__)

# To control caching so as to save and retrieve plot figs on client side


@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/insertintotable', methods=['POST'])
def insertintotable():
    nm = request.form['nm']

    # **************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2, end.month, end.day)
        data = yf.download(quote, start=start, end=end)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if (df.empty):
            ts = TimeSeries(key='OFLH89EQ9CYXCXGM', output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(
                symbol='NSE:'+quote, outputsize='full')
            # Format df
            # Last 2 yrs rows => 502, in ascending order => ::-1
            data = data.head(503).iloc[::-1]
            data = data.reset_index()
            # Keep Required cols only
            df = pd.DataFrame()
            df['Date'] = data['date']
            df['Open'] = data['1. open']
            df['High'] = data['2. high']
            df['Low'] = data['3. low']
            df['Close'] = data['4. close']
            df['Adj Close'] = data['5. adjusted close']
            df['Volume'] = data['6. volume']
            df.to_csv(''+quote+'.csv', index=False)
        return


if __name__ == '__main__':
    app.run()
