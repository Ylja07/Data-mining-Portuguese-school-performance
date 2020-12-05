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


def random_forest(x_placeholder, y_placeholder, **kwarg):
    clf = RandomForestClassifier(kwargs)
    clf.fit(x_placeholder, y_placeholder)


def main():
    prep_data()
    X = dataset.loc[:, dataset.columns != 'G3']
    y = dataset['G3']
    Pruning.random_forest_hyper_parameters(X, y)

main()
