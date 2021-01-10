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
from sklearn.inspection import permutation_importance


# load data
dataset = pd.read_csv("../data/student-por.csv", sep=";")
dataset2 = pd.read_csv("../data/student-mat.csv", sep=";")


def random_forest(x_placeholder, y_placeholder, **kwarg):
    clf = RandomForestClassifier(**kwarg)
    clf.fit(x_placeholder, y_placeholder)

    # importances = clf.feature_importances_
    # indices = np.argsort(importances)[::-1]
    #
    # # Print the feature ranking
    # print("Feature ranking:")
    #
    # for f in range(x_placeholder.shape[1]):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

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


def prep_multi_run(basic_G3=True):
    """
    Prepare the data for the runs.
    If this isn't done it will crash because of multiple attempts to change values.
    """
    data_por = DataPreparation(dataset)
    por = data_por.set_feature_values(basic_G3)
    data_math = DataPreparation(dataset2)
    math = data_math.set_feature_values(basic_G3)

    return por, math


def permutation(mlp, X_test, y_test):
    """
    Get the feature importance of the MLP.
    """
    result = permutation_importance(mlp, X_test, y_test, n_repeats=10,
                                    random_state=42, n_jobs=2)
    sorted_idx = result.importances_mean.argsort()

    fig, ax = plt.subplots()
    ax.boxplot(result.importances[sorted_idx].T,
               vert=False, labels=X_test.columns[sorted_idx])
    ax.set_title("Permutation Importances (test set)")
    fig.tight_layout()
    plt.show()


def run(dataset_por, dataset_math, exclude, name):
    # Prep data for usage.
    X = dataset_por.drop(exclude, axis=1)
    y = dataset_por['G3']

    X2 = dataset_math.drop(exclude, axis=1)
    y2 = dataset_math['G3']

    X_train, X_test, y_train, y_test = split_data(X, y, 0.2)

    pruning = Pruning(scoring='roc_auc', X=X_train, y=y_train)

    # forest_kargs = pruning.random_forest_hyper_parameters()
    # forest = random_forest(X_train, y_train, **forest_kargs)

    mlp_kargs = pruning.mlp_hyper_parameters()
    mlp = neural(X_train, y_train, **mlp_kargs)

    # permutation(mlp, X_test, y_test)

    # Get accuracy
    # predict_score(forest, X_test, X2, y_test, y2, "Random Forest " + name)
    predict_score(mlp, X_test, X2, y_test, y2, "MLP " + name)


def main():
    portuguese, Math = prep_multi_run()
    run(portuguese, Math, ['G3'], 'with G1 and G2')
    run(portuguese, Math, ['G2', 'G3'], 'with G1 and no G2')
    run(portuguese, Math, ['G1', 'G2', 'G3'], 'with no G1 and G2')


main()
