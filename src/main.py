import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from Pruning import Pruning
from DataPreparation import DataPreparation

# load data
dataset = pd.read_csv("../data/student-por.csv", sep=";")
dataset2 = pd.read_csv("../data/student-mat.csv", sep=";")


def random_forest(x_placeholder, y_placeholder, **kwarg):
    clf = RandomForestClassifier(**kwarg)
    clf.fit(x_placeholder, y_placeholder)
    return clf


def split_data(x_data, y_data, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
    return X_train, X_test, y_train, y_test


def main():
    # Prep data for usage.
    data = DataPreparation(dataset)
    por = data.set_feature_values()
    X = por.loc[:, por.columns != 'G3']
    y = por['G3']

    data2 = DataPreparation(dataset2)
    mat = data2.set_feature_values()
    X2 = mat.loc[:, mat.columns != 'G3']
    y2 = mat['G3']

    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    # TODO: 2 times cross-validation
    kargs = Pruning.random_forest_hyper_parameters(X_train, y_train)
    forest = random_forest(X_train, y_train, **kargs)

    # Get accuracy
    predict_list = forest.predict(X_test)
    accuracy = accuracy_score(y_test, predict_list)
    predict_list2 = forest.predict(X2)
    accuracy2 = accuracy_score(y2, predict_list2)
    print("trained set(por): " + str(accuracy))
    print("math set: " + str(accuracy2))


main()
