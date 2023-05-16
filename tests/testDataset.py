# Importing the libraries

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

import pandas as pd

from regressionFunction import selectError


def main():
    df = pd.read_csv('/Datasets/brooklyn_listings.csv')

    # for this example, we're going to estimate the price with sqft, bathroom, and bedrooms
    df = df[['price', 'bathrooms', 'sqft']].dropna()

    # show some random lines from our data
    print(df.sample(n=15))

    error1 = testFunction(df)
    error2 = selectError(df, 2)


    print("*********************************************************************************")
    print("error 1:")
    print(error1)

    print("error 2:")
    print(error2)




def testFunction(df):

    # seperate out our x and y values
    x_values = df[['bathrooms', 'sqft']].values
    y_values = df['price'].values

    # visual
    print(x_values[0], y_values[0])

    # define our polynomial model, with whatever degree we want
    degree = 2

    # PolynomialFeatures will create a new matrix consisting of all polynomial combinations
    # of the features with a degree less than or equal to the degree we just gave the model (2)
    poly_model = PolynomialFeatures(degree=degree)

    # transform out polynomial features
    poly_x_values = poly_model.fit_transform(x_values)

    # should be in the form [1, a, b, a^2, ab, b^2]
    print(f'initial values {x_values[0]}\nMapped to {poly_x_values[0]}')

    # let's fit the model
    poly_model.fit(poly_x_values, y_values)

    # we use linear regression as a base!!! ** sometimes misunderstood **
    regression_model = LinearRegression()

    regression_model.fit(poly_x_values, y_values)

    y_pred = regression_model.predict(poly_x_values)

    regression_model.coef_

    mean_error = mean_squared_error(y_values, y_pred, squared=False)

    # check our accuracy for each degree, the lower the error the better!
    number_degrees = [1, 2, 3, 4, 5, 6, 7]
    plt_mean_squared_error = []
    for degree in number_degrees:
        poly_model = PolynomialFeatures(degree=degree)

        poly_x_values = poly_model.fit_transform(x_values)
        poly_model.fit(poly_x_values, y_values)

        regression_model = LinearRegression()
        regression_model.fit(poly_x_values, y_values)
        y_pred = regression_model.predict(poly_x_values)

        plt_mean_squared_error.append(mean_squared_error(y_values, y_pred, squared=False))

    plt.scatter(number_degrees, plt_mean_squared_error, color="green")
    plt.plot(number_degrees, plt_mean_squared_error, color="red")
    plt.show()

    return mean_error












main()