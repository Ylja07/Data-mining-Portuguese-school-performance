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
import concurrent.futures


class Pruning:
    def __init__(self, scoring, X, y):
        self.scoring = scoring
        self.X = X
        self.y = y

    def random_forest_hyper_parameters(self):
        param_grid = {'max_depth': np.arange(3, 10), 'min_samples_split': [10, 15, 25, 35, 50, 75],
                      'min_samples_leaf': [5, 10, 15, 20, 25]}

        # Think about using other ways to rate the data instead of the standard accuracy.
        random_forest_grid = GridSearchCV(RandomForestClassifier(), param_grid, scoring=self.scoring).fit(self.X, self.y)

        print('Best params for random forrest are: ' + str(random_forest_grid.best_params_))

        return random_forest_grid.best_params_

    def mlp_hyper_parameters(self):
        best_mlp = None
        # Multithreading awesomeness 😎
        with concurrent.futures.ThreadPoolExecutor() as executor:
            mlp_adam = executor.submit(self.__adam)
            mlp_sgd = executor.submit(self.__sgd)
            mlp_lbfgs = executor.submit(self.__lbfgs)
            mlp_adam_result = mlp_adam.result()
            mlp_sgd_result = mlp_sgd.result()
            mlp_lbfgs_result = mlp_lbfgs.result()

        all_mlp = [mlp_lbfgs_result, mlp_sgd_result, mlp_adam_result]
        previous = 0

        for mlp in all_mlp:
            if mlp.best_score_ > previous:
                best_mlp = mlp
                previous = mlp.best_score_

        print('Best params for mlp are: ' + str(best_mlp.best_params_))

        return best_mlp.best_params_

    def __adam(self):
        adam = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'max_iter': [9000],
                'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,),
                                       (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)],
                'solver': ['adam']}
        return GridSearchCV(MLPClassifier(), adam, scoring=self.scoring).fit(self.X, self.y)

    def __sgd(self):
        sgd = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'max_iter': [9000],
               'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,),
                                      (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)],
               'solver': ['sgd'], 'learning_rate': ['constant', 'invscaling', 'adaptive']}
        return GridSearchCV(MLPClassifier(), sgd, scoring=self.scoring).fit(self.X, self.y)

    def __lbfgs(self):
        lbfgs = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'max_iter': [9000],
                 'hidden_layer_sizes': [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12,),
                                        (13,), (14,), (15,), (16,), (17,), (18,), (19,), (20,), (21,)],
                 'solver': ['lbfgs']}
        return GridSearchCV(MLPClassifier(), lbfgs, scoring=self.scoring).fit(self.X, self.y)
