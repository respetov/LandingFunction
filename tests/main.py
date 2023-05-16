# This is a sample Python script.
import re
import time

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import datetime

import pandas as pd
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    pd.set_option('display.max_columns', None)
    # pd.reset_option(“display.max_columns”)

    # Importing the dataset
    datas = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test3.csv')
    print(datas.columns)
    print(datas.head())
    datas = removeStatusCode(datas)
    print('primero despues de la limpia++++++++++++++++++++')
    print(datas.columns)
    print(datas.head())
    print('*****************************************')

    datas1 = pd.read_csv('/Users/jrodriguez/IdeaProjects/configuration extractor/Resources/csv/test4.csv')
    datas1 = removeStatusCode(datas1)
    print('segundo despues de la limpia++++++++++++++++++++')
    print(datas1.head())

    print('----------------------------------------')

    result = pd.concat([datas, datas1], axis=0)
    print(result.head())
    print(result.columns)
    print(result.size)







def removeStatusCode(datas):

    valid = datas.loc[datas[' response'] != 504]
    errors = datas.loc[datas[' response'] == 504]
    valid = valid.drop(columns=[' response', ' time in milliseconds'])

    print("errores:")
    print(datas.size)
    print(valid.size)
    print(errors.size)

    return valid



main()
