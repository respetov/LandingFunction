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
from sklearn.feature_selection import mutual_info_regression
import seaborn as sns


import pandas as pd


def dataExtractor():
    # First dataset, full range in params, almost all result are 0 (incorrectly formed)
    datas = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test.csv')
    datas = columnConverter(datas)

    # Second dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas1 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test1.csv')
    datas1 = removeStatusCode(datas1)

    # Third dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas2 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test2.csv')
    datas2 = removeStatusCode(datas2)

    # Fourth dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas3 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test3.csv')
    datas3 = removeStatusCode(datas3)

    # Fifth dataset, params delimited: years between 2019 - 2023, publications and collaborations: 100
    datas4 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test4.csv')
    datas4 = removeStatusCode(datas4)

    # Sixth dataset, years in full range, publications and collaborations delimited: 100
    datas5 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test5.csv')
    datas5 = removeStatusCode(datas5)

    # Seventh dataset, years in full range, publications and collaborations delimited: 100
    datas6 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test6.csv')
    datas6 = removeStatusCode(datas6)

    # eighth dataset, full range in params, almost all result are 0
    datas7 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test7.csv')
    datas7 = removeStatusCode(datas7)

    #result = pd.concat([datas], axis=0)
    result = pd.concat([datas, datas1, datas2, datas3, datas4, datas5, datas6, datas7], axis=0)

    #mi_score(result)

    result = makeResultsInteger(result)

    result = dateConverter(result)


    result = remove0(result)

    result = shuffle(result)
    result.reset_index(inplace=True, drop=True)

    #result = deleteDuplicates(result)

    printDatasMetrics(result)

    # result.to_csv('/Users/jrodriguez/PycharmProjects/LandingFunction/Datasets/datasetRefined')

    result1 = normalizeData(result)

    print("MI scores without normalize")
    mi_score(result)
    print("MI scores normalized")
    mi_score(result1)

    #printDatasMetricsNormalized(result)

    # Write dataset to csv (normalized)
    # result.to_csv('/Users/jrodriguez/PycharmProjects/LandingFunction/Datasets/datasetRefinedNormalized')

    return result1, result

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

    """print("errors estatus:")
    print(datas.size)
    print(valid.size)
    print(errors.size)"""

    return valid


# remove rows that has value 0
def remove0(datas):

    valid = datas.loc[datas[' Total count'] != 0]
    zeros = datas.loc[datas[' Total count'] == 0]

    print("zeros in dataset:")
    print(datas.head())
    print(len(datas.index))
    print(len(valid.index))
    print(len(zeros.index))

    zeros = shuffle(zeros)
    zeros.reset_index(inplace=True, drop=True)
    zeros = zeros.iloc[:4000]

    valid = pd.concat([valid, zeros])
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
    #datas[' delta'] = datas[' max date'] - datas[' min date']

    """print(datas.columns)
    print(datas.sample(n=5))"""

    return datas


# Convert string values into integers
def makeResultsInteger(datas):

    datas["publications"] = datas['publications'].apply(
        lambda x: int(x))
    datas[" collaborations"] = datas[' collaborations'].apply(
        lambda x: int(x))
    datas[" Total count"] = datas[' Total count'].apply(
        lambda x: int(x))

    return datas


# Normalize data betwen 0 and 1
def normalizeData(dataset):
    maxPublication = 410
    maxCollaboration = 3584
    minDate = 19460101
    maxDate = 20221231
    maxDiff = 761130

    result = dataset.copy()

    result["publications"] = result['publications'].apply(
        lambda x: (x - 1) / (maxPublication - 1))
    result[" collaborations"] = result[' collaborations'].apply(
        lambda x: (x - 1) / (maxCollaboration - 1))
    result[" min date"] = result[' min date'].apply(
        lambda x: (x - minDate) / (maxDate - minDate))
    result[" max date"] = result[' max date'].apply(
        lambda x: (x - minDate) / (maxDate - minDate))
    """result[" delta"] = result[' delta'].apply(
        lambda x: (x - 1) / (maxDiff - 1))"""
    return result



def printDatasMetrics(result):

    """plt.scatter(result["publications"], result[" Total count"], s=0.5)
    plt.xlim(0, 100)
    plt.ylim(0, 3000)
    plt.title('publications vs total count')
    plt.xlabel("publications")
    plt.ylabel("total count")
    plt.savefig("/Users/jrodriguez/Documents/models/Normalized/errores/manualy/pub_count_bounded", dpi=500)
    plt.show()

    plt.scatter(result[" collaborations"], result[" Total count"], s=0.5)
    plt.xlim(0, 1000)
    plt.ylim(0, 1500)
    plt.title('collaborations vs total count')
    plt.xlabel("collaborations")
    plt.ylabel("total count")
    plt.savefig("/Users/jrodriguez/Documents/models/Normalized/errores/manualy/coll_count_bounded", dpi=500)
    plt.show()

    plt.scatter(result[" min date"], result[" Total count"], s=0.5)
    plt.xlim(20150000, 20230000)
    plt.ylim(0, 1000)
    plt.title('min date vs total count')
    plt.xlabel("min date")
    plt.ylabel("total count")
    # plt.savefig("/Users/jrodriguez/Documents/models/Normalized/errores/manualy/minDate_count", dpi=500)
    plt.show()

    plt.scatter(result[" max date"], result[" Total count"], s=0.5)
    plt.xlim(20150000, 20230000)
    plt.ylim(0, 5000)
    plt.title('max date vs total count')
    plt.xlabel("max date")
    plt.ylabel("total count")
    # plt.savefig("/Users/jrodriguez/Documents/models/Normalized/errores/manualy/maxDate_count", dpi=500)
    plt.show()

    plt.scatter(result[" delta"], result[" Total count"], s=0.5)
    plt.title('delta vs total count')
    plt.xlabel("delta")
    plt.ylabel("total count")
    # plt.savefig("/Users/jrodriguez/Documents/models/Normalized/errores/manualy/delta_count", dpi=500)
    plt.show()"""


    """a = 1 / result["publications"]

    plt.scatter(a, result[" Total count"], s=0.5)
    #plt.xlim(0, 10)
    #plt.ylim(0, 1000)
    plt.title('1/publications')
    plt.show()

    b = 1 / result[" collaborations"]

    plt.scatter(b, result[" Total count"], s=0.5)
    #plt.xlim(0, 10)
    #plt.ylim(0, 1000)
    plt.title('1/collaborations')

    plt.show()"""

    # X, y = result[["publications", " collaborations", " min date", " max date"]], result[" Total count"].array

    plt.scatter(result["publications"], result[" Total count"], color='red')
    plt.title('publications vs result of the query')
    plt.xlabel("publications")
    plt.ylabel("Total count")
    plt.show()
    result["publications"].hist()
    plt.title('number of samples with certain number of publications')
    plt.xlabel("publications")
    plt.ylabel("number of samples")
    plt.show()

    plt.scatter(result[" collaborations"], result[" Total count"], color='red')
    plt.title('collaborations vs result of the query')
    plt.xlabel("collaborations")
    plt.ylabel("Total count")
    plt.show()
    result[" collaborations"].hist()
    plt.title('number of samples with certain number of collaborations')
    plt.xlabel("collaborations")
    plt.ylabel("number of samples")
    plt.show()

    plt.scatter(result[" min date"], result[" Total count"], color='red')
    plt.title('number of samples with certain number of min date')
    plt.show()
    result[" min date"].hist()
    plt.title('number of samples with certain number of min date')
    plt.xlabel("min date")
    plt.ylabel("total count")
    plt.show()

    plt.scatter(result[" max date"], result[" Total count"], color='red')
    plt.title('number of samples with certain number of max date')
    plt.show()
    result[" max date"].hist()
    plt.title('number of samples with certain number of max date')
    plt.xlabel("max date")
    plt.ylabel("total count")
    plt.show()

    """listrange = list(range(0, 100, 1))

    result[" Total count"].hist(bins=listrange)
    plt.title('total count')
    plt.show()"""

    # print(result[" Total count"].value_counts())

    """plt.scatter(X[" collaborations"], y, color='red')
    plt.title(' collaborations')
    plt.show()

    plt.scatter(X[" min date"], y, color='red')
    plt.title(' min date')
    plt.show()

    plt.scatter(X[" max date"], y, color='red')
    plt.title(' max date')
    plt.show()"""


def printDatasMetricsNormalized(result):
    c = result[' min date'].apply(
        lambda x: pow(math.e, x), 5)
    print(c)

    plt.scatter(c, result[" Total count"], s=0.5)
    """plt.xlim(0, 10)
    plt.ylim(0, 1000)"""
    plt.title('e ^ min date')
    plt.show()

    d = result[' max date'].apply(
        lambda x: pow(math.e, x), 5)
    print(d)

    plt.scatter(d, result[" Total count"], s=0.5)
    """plt.xlim(0, 10)
    plt.ylim(0, 1000)"""
    plt.title('e ^ max date')
    plt.show()

def deleteDuplicates(datas):
    pd.set_option('display.max_columns', None)

    datas = datas.sort_values(" min date", ascending=False)

    datas = datas[["publications", " collaborations", " min date", " max date", " Total count"]]

    """print("con duplicados:")
    print(datas.size)"""

    datas = datas.drop_duplicates(subset=["publications", " collaborations"])
    """print("sin duplicados")
    print(datas.size)
    print(datas.head)"""

    return datas

def mi_score(datas):

    X = datas[["publications", " collaborations", " min date", " max date", " Total count"]].copy()
    y = X.pop(" Total count")

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    #print(discrete_features)

    mi_scores = make_mi_scores(X, y, discrete_features)
    mi_scores[::3]
    print(mi_scores)

    """plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)

    X, y = datas[["publications", " collaborations", " min date", " max date",]], datas[" Total count"].array
    plt.scatter(X["publications"], y)
    plt.show()
    plt.scatter(X[" collaborations"], y)
    plt.show()"""


def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y) #, discrete_features=discrete_features
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()

