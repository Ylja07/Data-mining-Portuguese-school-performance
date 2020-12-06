import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pruning import Pruning

# load data
dataset = pd.read_csv("../data/student-por.csv", sep=";")

holdout_data = []


def prep_data():
    """
    This prepares our data set.
    at this moment it's programmed statically because i couldn't
    be fucked to do it with loops :)
    :rtype: object
    """
    dataset['school'].replace({'GP': 0, 'MS': 1}, inplace=True)
    dataset['sex'].replace({'F': 0, 'M': 1}, inplace=True)
    dataset['address'].replace({'U': 0, 'R': 1}, inplace=True)
    dataset['famsize'].replace({'LE3': 0, 'GT3': 1}, inplace=True)
    dataset['Pstatus'].replace({'T': 0, 'A': 1}, inplace=True)
    dataset['Mjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}, inplace=True)
    dataset['Fjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}, inplace=True)
    dataset['reason'].replace({'home': 0, 'reputation': 1, 'course': 2, 'other': 3}, inplace=True)
    dataset['guardian'].replace({'mother': 0, 'father': 1, 'other': 2}, inplace=True)
    dataset['schoolsup'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['famsup'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['paid'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['activities'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['nursery'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['higher'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['internet'].replace({'no': 0, 'yes': 1}, inplace=True)
    dataset['romantic'].replace({'no': 0, 'yes': 1}, inplace=True)
    # TODO: change G3 to binary passed or not (need to verify)
    dataset['G3'].replace({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
                           10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1, 17: 1, 18: 1, 19: 1, 20: 1}, inplace=True)


def random_forest(x_placeholder, y_placeholder, **kwarg):
    clf = RandomForestClassifier(**kwarg)
    clf.fit(x_placeholder, y_placeholder)
    return clf

def split_data(x_data, y_data, test_size):
    X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size)
    return X_train, X_test, y_train, y_test


def main():
    # Prep data for usage.
    prep_data()
    X = dataset.loc[:, dataset.columns != 'G3']
    y = dataset['G3']
    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    # Tune the hyper parameters & fit the forrest.
    kargs = Pruning.random_forest_hyper_parameters(X_train, y_train)
    forest = random_forest(X_train, y_train, **kargs)

    # Get accuracy
    predict_list = forest.predict(X_test)
    accuracy = accuracy_score(y_test, predict_list)
    print(accuracy)


main()
