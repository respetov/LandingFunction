import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt


def main():
    pd.set_option('display.max_columns', None)

    lection3()



#Creating Features
"""Tips on Creating Features
It's good to keep in mind your model's own strengths and weaknesses when creating features. Here are some guidelines:
Linear models learn sums and differences naturally, but can't learn anything more complex.
Ratios seem to be difficult for most models to learn. Ratio combinations often lead to some easy performance gains.
Linear models and neural nets generally do better with normalized features. Neural nets especially need features scaled to values not too far from 0. Tree-based models (like random forests and XGBoost) can sometimes benefit from normalization, but usually much less so.
Tree models can learn to approximate almost any combination of features, but when a combination is especially important they can still benefit from having it explicitly created, especially when data is limited.
Counts are especially helpful for tree models, since these models don't have a natural way of aggregating information across many features at once."""
def lection3():

    #plt.style.use("seaborn-whitegrid")
    plt.rc("figure", autolayout=True)
    plt.rc(
        "axes",
        labelweight="bold",
        labelsize="large",
        titleweight="bold",
        titlesize=14,
        titlepad=10,
    )

    accidents = pd.read_csv("/Datasets/1edition/accidents.csv")
    autos = pd.read_csv("/Datasets/1edition/autos.csv")
    concrete = pd.read_csv("/Datasets/1edition/concrete.csv")
    customer = pd.read_csv("/Datasets/1edition/customer.csv")

    autos["stroke_ratio"] = autos.stroke / autos.bore

    #print(autos[["stroke", "bore", "stroke_ratio"]].head())

    autos["displacement"] = (
            np.pi * ((0.5 * autos.bore) ** 2) * autos.stroke * autos.num_of_cylinders
    )

    # If the feature has 0.0 values, use np.log1p (log(1+x)) instead of np.log
    accidents["LogWindSpeed"] = accidents.WindSpeed.apply(np.log1p)

    # Plot a comparison
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
    sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1]);
    #plt.show()

    roadway_features = ["Amenity", "Bump", "Crossing", "GiveWay",
                        "Junction", "NoExit", "Railway", "Roundabout", "Station", "Stop",
                        "TrafficCalming", "TrafficSignal"]
    accidents["RoadwayFeatures"] = accidents[roadway_features].sum(axis=1)

    #print(accidents[roadway_features + ["RoadwayFeatures"]].head(10))

    components = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water",
                  "Superplasticizer", "CoarseAggregate", "FineAggregate"]
    concrete["Components"] = concrete[components].gt(0).sum(axis=1)

    #print(concrete[components + ["Components"]].head(10))

    customer[["Type", "Level"]] = (  # Create two new features
        customer["Policy"]  # from the Policy feature
        .str  # through the string accessor
        .split(" ", expand=True)  # by splitting on " "
        # and expanding the result into separate columns
    )

    #print(customer[["Policy", "Type", "Level"]].head(10))

    autos["make_and_style"] = autos["make"] + "_" + autos["body_style"]
    #print(autos[["make", "body_style", "make_and_style"]].head())

    customer["AverageIncome"] = (
        customer.groupby("State")  # for each state
        ["Income"]  # select the income
        .transform("mean")  # and compute its mean
    )

    #print(customer[["State", "Income", "AverageIncome"]].head(10))

    customer["StateFreq"] = (
            customer.groupby("State")
            ["State"]
            .transform("count")
            / customer.State.count()
    )

    print(customer[["State", "StateFreq"]].head(10))



#Mutual Information
def lection2():

    df = pd.read_csv("/Datasets/1edition/Automobile_data.csv")
    print(df.head())
    df = df.drop("normalized-losses", axis=1)
    print(df.head())

    df = cleanDf(df)

    X = df.copy()
    y = X.pop("price")

    # Label encoding for categoricals
    for colname in X.select_dtypes("object"):
        X[colname], _ = X[colname].factorize()

    # All discrete features should now have integer dtypes (double-check this before using MI!)
    discrete_features = X.dtypes == int

    print(discrete_features)

    mi_scores = make_mi_scores(X, y, discrete_features)
    mi_scores[::3]  # show a few features with their MI scores

    print(mi_scores)

    def plot_mi_scores(scores):
        scores = scores.sort_values(ascending=True)
        width = np.arange(len(scores))
        ticks = list(scores.index)
        plt.barh(width, scores)
        plt.yticks(width, ticks)
        plt.title("Mutual Information Scores")

    plt.figure(dpi=100, figsize=(8, 5))
    plot_mi_scores(mi_scores)
    plt.show()




def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_regression(X, y) #, discrete_features=discrete_features
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def cleanDf(df):
    for (columnName, columnData) in df.items():
        valid = df.loc[df[columnName] != "?"]
        df = valid
    return df




#What Is Feature Engineering
def lection1():
    df = pd.read_csv("/Datasets/1edition/concrete.csv")
    print(df.head())

    X = df.copy()
    y = X.pop("CompressiveStrength")

    # Train and score baseline model
    baseline = RandomForestRegressor(criterion="absolute_error", random_state=0)
    baseline_score = cross_val_score(
        baseline, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    baseline_score = -1 * baseline_score.mean()

    print(f"MAE Baseline Score: {baseline_score:.4}")

    X = df.copy()
    y = X.pop("CompressiveStrength")

    # Create synthetic features
    X["FCRatio"] = X["FineAggregate"] / X["CoarseAggregate"]
    X["AggCmtRatio"] = (X["CoarseAggregate"] + X["FineAggregate"]) / X["Cement"]
    X["WtrCmtRatio"] = X["Water"] / X["Cement"]

    # Train and score model on dataset with additional ratio features
    model = RandomForestRegressor(criterion="absolute_error", random_state=0)
    score = cross_val_score(
        model, X, y, cv=5, scoring="neg_mean_absolute_error"
    )
    score = -1 * score.mean()

    print(f"MAE Score with Ratio Features: {score:.4}")




main()