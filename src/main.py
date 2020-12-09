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
dataset2 = pd.read_csv("../data/student-mat.csv", sep=";")


def prep_data(data_set):
    """
    This prepares our data set.
    at this moment it's programmed statically because i couldn't
    be fucked to do it with loops :)
    :rtype: object
    """
    data_set['school'].replace({'GP': 0, 'MS': 1}, inplace=True)
    data_set['sex'].replace({'F': 0, 'M': 1}, inplace=True)
    data_set['address'].replace({'U': 0, 'R': 1}, inplace=True)
    data_set['famsize'].replace({'LE3': 0, 'GT3': 1}, inplace=True)
    data_set['Pstatus'].replace({'T': 0, 'A': 1}, inplace=True)
    data_set['Mjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}, inplace=True)
    data_set['Fjob'].replace({'teacher': 0, 'health': 1, 'services': 2, 'at_home': 3, 'other': 4}, inplace=True)
    data_set['reason'].replace({'home': 0, 'reputation': 1, 'course': 2, 'other': 3}, inplace=True)
    data_set['guardian'].replace({'mother': 0, 'father': 1, 'other': 2}, inplace=True)
    data_set['schoolsup'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['famsup'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['paid'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['activities'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['nursery'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['higher'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['internet'].replace({'no': 0, 'yes': 1}, inplace=True)
    data_set['romantic'].replace({'no': 0, 'yes': 1}, inplace=True)
    # TODO:  Possibly 4 numerical values to see which students barely passed any see why.
    data_set['G3'].replace({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0,
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
    prep_data(dataset)
    X = dataset.loc[:, dataset.columns != 'G3']
    y = dataset['G3']

    prep_data(dataset2)
    X2 = dataset2.loc[:, dataset2.columns != 'G3']
    y2 = dataset2['G3']

    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    # Tune the hyper parameters & fit the forrest.
    # TODO: 2 times cross-validation
    # TODO: Possibly make 2 models (1 with G1,G2 and one without)
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
