


def main():

    """X = datas.iloc[:, 0:4].values
    y = datas.iloc[:, 4].values

    print(X)
    print("-----------")
    print(y)"""

    """lin = LinearRegression()

    lin.fit(X, y)


    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    poly.fit(X_poly, y)
    lin2 = LinearRegression()
    lin2.fit(X_poly, y)"""

    #print(lin2.predict(poly.fit_transform(X)))

    """
    # Visualising the Linear Regression results
    plt.scatter(X, y, color='blue')

    plt.plot(X, lin.predict(X), color='red')
    plt.title('Linear Regression')

    plt.show()
    """

    # Visualising the Polynomial Regression results
    """plt.scatter(X, y, color='blue')

    plt.plot(X, lin2.predict(poly.fit_transform(X)), color='red')
    plt.title('Polynomial Regression')

    plt.show()"""

    # Predicting a new result with Linear Regression after converting predict variable to 2D array
    """pred = 1673259201
    predarray = np.array([pred])
    result = lin.predict(predarray)"""

    # Predicting a new result with Polynomial Regression after converting predict variable to 2D array
    """pred2 = [34, 71, 1627776000000, 1665705600000]
    pred2array = np.array([pred2])
    result2 = lin2.predict(poly.fit_transform(pred2array))

    print(result2)"""