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
import sklearn
import plotly.express as px
from sklearn.utils.fixes import sp_version, parse_version
from sklearn.linear_model import QuantileRegressor


import pandas as pd

from DataExtractor import dataExtractor
from LinearRegression import linearRegression
from QuantileRegression import quantileRegression


def main():

    datasNormalized, datasNoNorm = dataExtractor()

    print(datasNormalized.head())
    print(datasNoNorm.head())
    print("********************************************************")

    """fig = px.scatter_matrix(datasNoNorm)
    fig.show()"""

    selectLinearError(datasNormalized, datasNoNorm)

    #selectQuantileError(datas)


def selectLinearError(datasNormalized, datasNoNorm):
    dictionary = {}

    valid = {}
    invalid = {}

    #datas = deleteDuplicates(datas)

    for x in range(1, 2):
        error, val, inval = linearRegression(datasNormalized, datasNoNorm, x)
        dictionary[x] = error
        valid[x] = val
        invalid[x] = inval
        print(x)
    print(dictionary)
    print(valid)
    print(invalid)

    plt.plot(dictionary.keys(), dictionary.values(), color='red')
    plt.title('polynomial Regression')
    plt.show()

    plt.plot(valid.values(), invalid.values(), color='red')
    plt.title('errors between -20 and 20 and outsiders')
    plt.show()

    datasNormalized.hist()
    plt.show()


def selectQuantileError(datas):
    error = quantileRegression(datas)








main()
