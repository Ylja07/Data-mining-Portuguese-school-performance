import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
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


def neural(x_placeholder, y_placeholder, **kwarg):
    mlp = MLPClassifier(**kwarg)
    mlp.fit(x_placeholder, y_placeholder)
    return mlp


def predict_score(clf, x1, x2, y1, y2, clf_identifier):
    predict_list = clf.predict(x1)
    accuracy = accuracy_score(y1, predict_list)
    predict_list2 = clf.predict(x2)
    accuracy2 = accuracy_score(y2, predict_list2)
    print(clf_identifier + " trained set(por): " + str(accuracy))
    print(clf_identifier + " math set: " + str(accuracy2))


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

    pruning = Pruning(cv=None, X=X_train, y=y_train)

    forest_kargs = pruning.random_forest_hyper_parameters()
    forest = random_forest(X_train, y_train, **forest_kargs)

    mlp_kargs = pruning.mlp_hyper_parameters()
    mlp = neural(X_train, y_train, **mlp_kargs)

    # Get accuracy
    predict_score(forest, X_test, X2, y_test, y2, "Random Forest")
    predict_score(mlp, X_test, X2, y_test, y2, "MLP")


main()
