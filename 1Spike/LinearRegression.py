# Importing the libraries
import math
import re
import time

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



def linearRegression(datas, datasNoNormalized, degree):
    np.set_printoptions(suppress=True)
    #datas = pd.read_csv('/Users/jrodriguez/PycharmProjects/LandingFunction/Datasets/data.csv')
    X, y = datas[["publications", " collaborations"]], datas[" Total count"] #, " min date", " max date"

    """X = datas.iloc[:, 1:2].values
    y = datas.iloc[:, 2].values"""

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    #poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #, random_state=42

    X_train = poly.fit_transform(X_train)
    X_test = poly.transform(X_test)


    poly.fit(X_train, y_train)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, y_train)

    importance = poly_reg_model.coef_

    """for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))"""
    # plot feature importance
    """plt.bar([x for x in range(len(importance))], importance)
    plt.show()"""

    # mymodel = np.poly1d(np.polyfit(X_train, y_train, degree))

    poly_reg_y_predicted = poly_reg_model.predict(X_test)

    # RMSE= Root mean square error
    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))

    valid, invalid = printFunctionMetrics(y_test, poly_reg_y_predicted, poly_reg_rmse, poly_reg_model, X_test, poly, datas, datasNoNormalized)


    return poly_reg_rmse, valid, invalid

def printFunctionMetrics(y_test, poly_reg_y_predicted, poly_reg_rmse, poly_reg_model, X_test, poly, datas, datasNoNormalized):
    # print(poly_reg_model.coef_)

    myline = np.linspace(1, 3000, 2000)

    # plt.scatter(x, y)
    # plt.plot(poly_reg_model.predict(X_test))
    # plt.show()

    """print("diferencia entre predict y original:")
    print(poly_reg_y_predicted[:10])
    print(y_test[:10])
    plt.scatter(poly_reg_y_predicted, y_test)
    plt.xlim(-100, 1500)
    plt.ylim(0, 1500)
    plt.show()"""

    """plt.scatter(X_test[:, [0]], y_test)
    plt.xlim(-100, 500)
    plt.ylim(0, 500)
    plt.plot(X_test[:, [0]], poly_reg_y_predicted, color='red')
    plt.show()"""

    errors = y_test - poly_reg_y_predicted

    cleanErrors = []
    for x in errors:
        if x < 0:
            x = x * -1
        cleanErrors.append(x)

    order = sorted(cleanErrors, reverse=True)

    print("first 10 errors elements order by error value")
    print(order[:10])

    first = max(errors)

    #errors.index[first]
    position = 0
    for x in errors:
        if first == x:
            break
        position = position + 1

    y_value = y_test.iloc[position]

    x_values = datasNoNormalized.loc[datasNoNormalized[' Total count'] == y_value]

    print("position ++++++++++++++++++++++++++")
    print(position)
    print("y_test, predicted, error and first:")
    print(y_value, " ,", poly_reg_y_predicted[position], " ,", errors.iloc[position], " ,", first)
    print("X_values:")
    print(x_values)



    valid_error = 0
    invalid_errors = 0
    for x in errors:
        if x >= -20 and x <= 20:
            valid_error = valid_error + 1
        else:
            invalid_errors = invalid_errors + 1

    percentile = np.percentile(cleanErrors, np.arange(10, 110, 10))
    #np.round(percentile)

    dic = {}
    for i, j in zip(percentile, np.arange(10, 110, 10)):
        dic[j] = np.sum(cleanErrors <= i)

    print("percentiles: ")
    print(percentile)
    """print("numbers of elements in each percentile: ")
    print(dic)
    for key, value in dic.items():
        print(str(value) + ' numbers in ' + str(key) + ' percentile')
    quantiles = np.linspace(1, len(errors), 11, dtype = np.int64)
    print(quantiles)"""


    """errorCounts.hist()
    plt.show()

    errors.hist()
    plt.show()"""
    """print("valid erors y invalid errors:")
    print(valid_error)
    print(invalid_errors)"""
    #print(errorCounts)
    #print(errors.size)

    # errors.hist()
    # print(np.histogram(errors))
    # plt.show()

    """listrange = list(range(0, 100, 5))
    a = np.hstack(errors)
    plt.hist(a, bins=listrange)
    plt.show()

    listrangeb = list(range(0, 1000, 20))
    plt.hist(a, bins=listrangeb)
    plt.show()

    plt.boxplot(errors)
    plt.show()"""

    """print("error: ")
    print(poly_reg_rmse)
    print(poly_reg_model.score(X_test, y_test))"""

    """predNatural = [34]  # , 71, 20190101, 30000, 20220101  - , 0.01953670, 0.95910028, 0.03941381, 0.99851536
    pred = [0.08068459]
    predarray = np.array([pred])
    result = poly_reg_model.predict(poly.fit_transform(predarray))

    print("Predict of: ")
    print(predNatural)
    print(result)

    pred2natural = [100]  # , 200, 20100101, 120000, 20220101 - , 0.05554005, 0.84085504, 0.15765921, 0.99851536
    pred2 = [0.24205378]
    pred2array = np.array([pred2])
    result2 = poly_reg_model.predict(poly.fit_transform(pred2array))

    print("Predict of: ")
    print(pred2natural)
    print(result2)

    pred3natual = [1]  # , 1, 20220101, 10000, 20230101 -, 0 , 0.84085504, 0.01313706, 0.99851536
    pred3 = [0]
    pred3array = np.array([pred3])
    result3 = poly_reg_model.predict(poly.fit_transform(pred3array))

    print("Predict of: ")
    print(pred3natual)
    print(result3)"""

    """X = datas[["publications"]]
    y = datas[" Total count"]
    X = datas.iloc[:, 1:2].values
    y = datas.iloc[:, 2].values
    poly_features = poly.fit_transform(X)
    y_pred = poly_reg_model.predict(poly_features)"""

    # Visualising the Linear Regression results
    """plt.scatter(X, y, color='blue')

    plt.plot(X, y_pred, color='red')
    plt.title('Linear Regression')

    plt.show()"""

    return valid_error, invalid_errors



