from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.offline import plot
import plotly.graph_objs as go
import chart_studio.plotly as py
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

with open('dataset.txt', 'r') as f:
    dataPath = f.readline()
f.close()
Dataset = pd.read_csv(dataPath)
Dataset = Dataset.dropna()
Dataset = Dataset[['Date', 'Open', 'High',
                   'Low', 'Close', 'Adj Close', 'Volume']]
Dataset.head()
Dataset.describe()
Dataset.info()

# init_notebook_mode(connected=True)
layout = go.Layout(
    title='STOCK PRICE OF Dataset',
    xaxis=dict(
        title='date',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='blue'
        )
    ),
    yaxis=dict(
        title='Price',
        titlefont=dict(
            family='Courier New, monospace',
            size=18,
            color='red'
        )
    )
)
Dataset_DATA = [{'x': Dataset['Date'], 'y':Dataset['Close']}]
plot = go.Figure(data=Dataset_DATA, layout=layout)
Dataset['Open-Close'] = Dataset.Close - Dataset.Open
Dataset['High-Low'] = Dataset.High - Dataset.Low
Dataset = Dataset.dropna()
X = Dataset[['Open-Close', 'High-Low']]
X.head()
Y = np.where(Dataset['Close'].shift(-1) > Dataset['Close'], 1, -1)
split_percentage = 0.8
split = int(split_percentage*len(Dataset))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]
scores = []

for num_trees in range(1, 41):
    clf = RandomForestClassifier(n_estimators=num_trees)
    scores.append(cross_val_score(clf, X, Y, cv=10))
print(scores[0])
rfc = RandomForestClassifier(n_estimators=16)
rfc.fit(X_train, Y_train)
RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=16,
                       n_jobs=None, oob_score=False, random_state=None,
                       verbose=0, warm_start=False)
rfc_pred = rfc.predict(X_test)
print(classification_report(Y_test, rfc_pred))
print(confusion_matrix(Y_test, rfc_pred))
Dataset['Predicted_Signal'] = rfc.predict(X)

Dataset['Actual'] = np.log(Dataset['Close']/Dataset['Close'].shift(1))
Cumulative_SPY_returns = Dataset[split:]['Actual'].cumsum()*100

Dataset['Predicted'] = Dataset['Actual'] * \
    Dataset['Predicted_Signal'].shift(1)
Cumulative_Strategy_returns = Dataset[split:]['Predicted'].cumsum()*100

plt.figure(figsize=(10, 5))
plt.plot(Cumulative_SPY_returns, color='r', label='Actual')
plt.plot(Cumulative_Strategy_returns, color='g', label='Predicted')
plt.legend()
# plt.show()
plt.savefig('templates/result.png')
Std = Cumulative_Strategy_returns.std()
Sharpe = (Cumulative_Strategy_returns-Cumulative_SPY_returns)/Std
Sharpe = Sharpe.mean()

model = rfc.fit(X_train, Y_train)
model = rfc.fit(X_train, Y_train)
probability = model.predict_proba(X_test)

predicted = rfc.predict(X_test)


print('Accuray')
print(model.score(X_train, Y_train))
ac = model.score(X_train, Y_train)
with open('templates/accuracy.txt', 'w') as f:

    f.write(str(ac))
