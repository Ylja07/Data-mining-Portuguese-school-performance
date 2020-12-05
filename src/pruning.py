import numpy as np
import pandas as pd
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


class Pruning:
    def random_forest_hyper_parameters(X, y):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        param_grid = {'max_depth': np.arange(3, 10), 'min_samples_split': [10, 25, 50, 75], 'min_samples_leaf': [5, 10, 15, 20, 25]}

        random_forest_grid = GridSearchCV(RandomForestClassifier(), param_grid)

        random_forest_grid.fit(X_train, y_train)

        print('Best params are: ' + str(random_forest_grid.best_params_))