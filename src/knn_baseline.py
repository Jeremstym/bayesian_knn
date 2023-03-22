from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, train_test_split
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay

# import some data to play with
iris = datasets.load_iris()

data = iris.data
target = iris.target

# ---------- With classical mean accuracy -------------

neigh = KNeighborsClassifier(n_neighbors=4)

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    target, 
                                                    test_size=0.33, 
                                                    random_state=42)

neigh.fit(X_train, y_train)
neigh.score(X_test, y_test)

neigh.predict(X_test)
neigh.predict_proba(X_test)

# ------------- Leave one out method ---------------------

loo = LeaveOneOut()
counter = 0

for train_index, test_index in loo.split(data):
    neigh = KNeighborsClassifier(n_neighbors=4)

    neigh.fit(data[train_index], target[train_index])
    counter += (neigh.predict(data[test_index]) == target[test_index])

metrics = counter / 150
metrics[0]