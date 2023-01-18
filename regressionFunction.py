# Importing the libraries
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

import pandas as pd


def main():

    datas = dataExtractor()
    dictionary = {}

    for x in range(1, 20):
        error = selectError(datas, x)
        dictionary[x] = error
        print(x)
    print(dictionary)

    plt.plot(dictionary.keys(), dictionary.values(), color='red')
    plt.title('polynomial Regression, datas and datas7')
    plt.show()

    datas.hist()
    plt.show()


def selectError(datas, degree):

    X, y = datas[["publications", " collaborations", " min date", " max date"]], datas[" Total count"].array

    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(poly_features, y, test_size=0.3, random_state=42)

    poly_reg_model = LinearRegression()
    poly_reg_model.fit(X_train, y_train)

    poly_reg_y_predicted = poly_reg_model.predict(X_test)

    # RMSE= Root mean square error
    poly_reg_rmse = np.sqrt(mean_squared_error(y_test, poly_reg_y_predicted))
    print("error: ")
    print(poly_reg_rmse)
    print(poly_reg_model.score(X_test, y_test))

    pred = [34, 71, 20190101, 20220101]
    predarray = np.array([pred])
    result = poly_reg_model.predict(poly.fit_transform(predarray))

    print("Predict of: ")
    print(pred)
    print(result)

    pred2 = [100, 200, 20100101, 20220101]
    pred2array = np.array([pred2])
    result2 = poly_reg_model.predict(poly.fit_transform(pred2array))

    print("Predict of: ")
    print(pred2)
    print(result2)

    pred2 = [1, 1, 20220101, 20230101]
    pred2array = np.array([pred2])
    result2 = poly_reg_model.predict(poly.fit_transform(pred2array))

    print("Predict of: ")
    print(pred2)
    print(result2)

    return poly_reg_rmse,


def dataExtractor():
    #First dataset, full range in params, almost all result are 0 (incorrectly formed)
    datas = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test.csv')
    datas = columnConverter(datas)

    #Second dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas1 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test1.csv')
    datas1 = removeStatusCode(datas1)
    
    #Third dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas2 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test2.csv')
    datas2 = removeStatusCode(datas2)

    #Fourth dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas3 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test3.csv')
    datas3 = removeStatusCode(datas3)

    #Fifth dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas4 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test4.csv')
    datas4 = removeStatusCode(datas4)

    #Sixth dataset, years in full range, publications and collaborations delimited: 100
    datas5 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test5.csv')
    datas5 = removeStatusCode(datas5)

    #Seventh dataset, years in full range, publications and collaborations delimited: 100
    datas6 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test6.csv')
    datas6 = removeStatusCode(datas6)

    #eighth dataset, full range in params, almost all result are 0
    datas7 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test7.csv')
    datas7 = removeStatusCode(datas7)

    #result = pd.concat([datas, datas7], axis=0)
    result = pd.concat([datas, datas1, datas2, datas3, datas4, datas5, datas6, datas7], axis=0)

    result = dateConverter(result)

    result = normalizeResults(result)

    #result = remove0(result)

    result = shuffle(result)
    result.reset_index(inplace=True, drop=True)

    return result

# Cleaning the first dataset that is not correclty formed
def columnConverter(datas):

    datas.columns = ['publications', ' collaborations', ' min date', ' max date', ' Total count',
                     ' time in milliseconds']
    datas['publications'] = [re.sub('publications:', '', str(x)) for x in datas['publications']]
    datas[' collaborations'] = [re.sub(' collaborations:', '', str(x)) for x in datas[' collaborations']]
    datas[' min date'] = [re.sub(' min date:', '', str(x)) for x in datas[' min date']]
    datas[' max date'] = [re.sub(' max date:', '', str(x)) for x in datas[' max date']]
    datas[' Total count'] = [re.sub(' Total count:', '', str(x)) for x in datas[' Total count']]
    datas[' time in milliseconds'] = [re.sub(' time in milliseconds:', '', str(x)) for x in
                                     datas[' time in milliseconds']]

    result = datas.drop(columns=[' time in milliseconds'])

    return result

def removeStatusCode(datas):

    valid = datas.loc[datas[' response'] != 504]
    errors = datas.loc[datas[' response'] == 504]
    valid = valid.drop(columns=[' response', ' time in milliseconds'])

    """print("errors:")
    print(datas.size)
    print(valid.size)
    print(errors.size)"""

    return valid

#remove rows that has value 0
def remove0(datas):

    valid = datas.loc[datas[' Total count'] != 0]
    zeros = datas.loc[datas[' Total count'] == 0]

    print("zeros in dataset:")
    print(datas.size)
    print(valid.size)
    print(zeros.size)

    return valid

# Converting date strings to dates and dates to timestamp
def dateConverter(datas):

    """datas[' min date'] = datas[' min date'].apply(
        lambda x: time.mktime(datetime.strptime(x, ' %Y-%m-%d').date().timetuple()))
    datas[' max date'] = datas[' max date'].apply(
        lambda x: time.mktime(datetime.strptime(x, ' %Y-%m-%d').date().timetuple()))"""

    datas[' min date'] = datas[' min date'].apply(
        lambda x: datetime.strptime(x, ' %Y-%m-%d').date()).apply(
        lambda x: (x.year * 10000) + (x.month * 100) + (x.day * 1))
    datas[' max date'] = datas[' max date'].apply(
        lambda x: datetime.strptime(x, ' %Y-%m-%d').date()).apply(
        lambda x: (x.year * 10000) + (x.month * 100) + (x.day * 1))

    return datas

#Convert string values into integers
def normalizeResults(datas):

    datas["publications"] = datas['publications'].apply(
        lambda x: int(x))
    datas[" collaborations"] = datas[' collaborations'].apply(
        lambda x: int(x))
    datas[" Total count"] = datas[' Total count'].apply(
        lambda x: int(x))

    return datas




main()
