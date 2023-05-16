# Importing the libraries
import math
import re
import time
import csv

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
import time


import pandas as pd


pd.set_option('display.max_columns', None)

def main():
    #datas = pd.read_csv('/Users/jrodriguez/Documents/models/2edition/ckg_kgraph_author_contribution_202305110817.csv')
    datas = pd.read_csv('/Users/jrodriguez/Documents/models/2edition/ckg_kgraph_author_contribution_202305111257.csv')

    #Parameters for execution:
    printMetrics = False
    extractDataset = True

    if printMetrics:
        print_metrics(datas)
    if extractDataset:
        data_extracting(datas)

def print_metrics(datas):

    cont = datas['contributions']
    coll = datas['collaborations']
    year = datas['pub_year']

    print(cont.max())
    print(coll.max())

    listrange = list(range(0, 100, 1))
    listrangeAge = list(range(1990, 2024, 1))
    cont.hist(bins=listrange)
    #plt.xlim(0, 200)
    plt.show()
    coll.hist(bins=listrange)
    #plt.xlim(0, 1000)
    plt.show()
    year.hist(bins=listrangeAge)
    #plt.xlim(1990, 2024)
    plt.show()


    metrics = datas[["pub_year", "contributions", "collaborations"]].describe()
    print(metrics)
    print(datas.head(1))

    year = datas.loc[datas['pub_year'] == 2010]
    print(datas.size)
    print(year.size)
    yearD = year.drop_duplicates(subset=["distinct_author_id"])
    print(yearD.size)
    #print(yearD)

    yearCon = year.loc[datas['contributions'] == 5]
    print(yearCon.size)
    #print(yearCon)

    print(yearCon[["collaborations"]].describe())


def data_extracting(datas):
    with open("Datasets/2edition/testDataset3", "w", newline='') as csvfile:
        fieldnames = ["pub_year", "contributions", "collaborations", "total_count", "valid_query"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        #result = pd.DataFrame(columns=["pub_year", "contributions", "collaborations", "total_count", "valid_query"])

        start = time.time()

        year = datas.loc[datas['pub_year'] == 2021]
        contributions = year['contributions'].max()

        print("contributions: ")
        print(contributions)

        for x in range(1, contributions):

            yearCon = year.loc[year['contributions'] >= x]

            yearConForMaximum = year.loc[year['contributions'] == x]
            collaborations = yearConForMaximum['collaborations'].max()

            print("contributions: ")
            print(x)
            print("collaborations: ")
            print(collaborations)

            if pd.isna(collaborations):
                total_count = yearCon.size
                if total_count > 400:
                    valid = False
                else:
                    valid = True

                writer.writerow({"pub_year": 2021, "contributions": x, "collaborations": 1,
                                 "total_count": total_count, "valid_query": valid})
            else:
                for y in range(1, collaborations):

                    yearColl = yearCon.loc[yearCon['collaborations'] >= y]
                    total_count = yearColl.size

                    if total_count > 400:
                        valid = False
                    else:
                        valid = True

                    writer.writerow({"pub_year": 2021, "contributions": x, "collaborations": y,
                                     "total_count": total_count, "valid_query": valid})
                    #result = result.append(new_row, ignore_index=True)
        end = time.time()
        print(end - start)


def drop_data_extracting(datas):
    with open("Datasets/2edition/testDataset2", "w", newline='') as csvfile:
        fieldnames = ["pub_year", "contributions", "collaborations", "total_count", "valid_query"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # result = pd.DataFrame(columns=["pub_year", "contributions", "collaborations", "total_count", "valid_query"])

        start = time.time()

        year = datas.loc[datas['pub_year'] == 2021]
        year = year.sort_values('contributions')
        year.reset_index(inplace=True, drop=True)
        contributions = year['contributions'].max()

        print("contributions: ")
        print(contributions)
        yearCon = year

        for x in range(1, contributions):
            print("+++++++++++++++++++++++++++++++++++++++++")
            print(yearCon.size)

            posCon = year[year['contributions'] == x].index.values

            if not np.any(posCon):
                total_count = yearCon.size
                if total_count > 400:
                    valid = False
                else:
                    valid = True
                '''new_row = {"pub_year": 2021, "contributions": x, "collaborations": 1, "total_count": 0}
                result.loc[result.size] = new_row'''
                writer.writerow({"pub_year": 2021, "contributions": x, "collaborations": 1,
                                 "total_count": total_count, "valid_query": valid})
            else:
                # yearCon = year.loc[year['contributions'] >= x]
                yearCon = year.iloc[posCon[0]:]
                # collaborations = yearCon['collaborations'].max()
                collaborations = yearCon['collaborations'].max()

                print("contributions: ")
                print(x)
                print("collaborations: ")
                print(collaborations)

                if pd.isna(collaborations):
                    total_count = yearCon.size
                    if total_count > 400:
                        valid = False
                    else:
                        valid = True
                    '''new_row = {"pub_year": 2021, "contributions": x, "collaborations": 1, "total_count": 0}
                    result.loc[result.size] = new_row'''
                    writer.writerow({"pub_year": 2021, "contributions": x, "collaborations": 1,
                                     "total_count": total_count, "valid_query": valid})
                else:
                    for y in range(1, collaborations):
                        yearCon = yearCon.sort_values('collaborations')
                        yearCon.reset_index(inplace=True, drop=True)

                        posColl = yearCon[yearCon['collaborations'] == y].index.values
                        # yearColl = yearCon.loc[yearCon['collaborations'] <= y]

                        if np.any(posColl):
                            yearColl = yearCon.iloc[posColl[0]:]
                        total_count = yearColl.size

                        if total_count > 400:
                            valid = False
                        else:
                            valid = True

                        writer.writerow({"pub_year": 2021, "contributions": x, "collaborations": y,
                                         "total_count": total_count, "valid_query": valid})
                        '''new_row = {"pub_year": 2021, "contributions": x, "collaborations": y,
                                   "total_count": total_count, "valid_query": valid}
                        result.loc[result.size] = new_row'''
                        # result = result.append(new_row, ignore_index=True)

    #print(result)
    end = time.time()
    print(end - start)

    #Write dataset to csv
    #result.to_csv('/Users/jrodriguez/PycharmProjects/LandingFunction/Datasets/2edition/testDataset2')






main()