import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype
from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from datetime import datetime
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import plotly.express as px
from sklearn.utils.fixes import sp_version, parse_version
from sklearn.linear_model import QuantileRegressor


import pandas as pd


def quantileRegression(datas):
    X, y = datas[["publications", " collaborations"]], datas[" Total count"].array #, " min date", " max date"

    solver = "highs" if sp_version >= parse_version("1.6.0") else "interior-point"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    quantiles = [0.1, 0.5, 0.9]
    out_bounds_predictions = np.zeros_like(y_test, dtype=np.bool_)
    for quantile in quantiles:
        qr = QuantileRegressor(quantile=quantile, alpha=0, solver=solver)
        model = qr.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if quantiles == min(quantiles):
            out_bounds_predictions = np.logical_or(out_bounds_predictions, y_pred >= y_test)
        elif quantile == max(quantiles):
            out_bounds_predictions = np.logical_or(out_bounds_predictions, y_pred <= y_test)

    mae = printQuantileFunctionMetrics(y_test, y_pred, model, qr)

    return mae

def printQuantileFunctionMetrics(y, y_pred, model, qr):

    MAE = mean_absolute_error(y, y_pred)
    print("quantile error:")
    print(MAE)

    print("y predict:")
    print(y_pred)

    print("y real")
    print(y)

    predNatural = [34, 71, 20190101, 20220101]  # ,  30000 - , 0.03941381
    pred = [0.08068459, 0.01953670, 0.95910028, 0.99851536]
    predarray = np.array([pred])
    result = model.predict(predarray)

    print("Predict of: ")
    print(predNatural)
    print(result)

    pred2natural = [100, 200, 20100101, 20220101]  # ,  120000- , 0.15765921
    pred2 = [0.24205378, 0.05554005, 0.84085504, 0.99851536]
    pred2array = np.array([pred2])
    result2 = model.predict(pred2array)

    print("Predict of: ")
    print(pred2natural)
    print(result2)

    pred3natual = [1, 1, 20220101, 20230101]  # ,  10000- , 0.01313706
    pred3 = [0, 0, 0.84085504, 0.99851536]
    pred3array = np.array([pred3])
    result3 = model.predict(pred3array)

    print("Predict of: ")
    print(pred3natual)
    print(result3)

    plt.scatter(y, y_pred, s=0.5)
    plt.xlim(0, 300)
    """plt.ylim(0, 1000)"""
    plt.title('y vs y predicted')
    plt.show()

    return MAE




