
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
# nltk.download('punkt')

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
    print(nm)
    with open('dataset.txt', 'w') as f:
        str1 = 'Dataset/stocks/' + nm + '.csv'
        print(str1)
        f.write(str1)

    # **************** FUNCTIONS TO FETCH DATA ***************************
    import main2
    return render_template('result.html')


if __name__ == '__main__':
    app.run()
