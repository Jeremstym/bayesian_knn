from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn.inspection import DecisionBoundaryDisplay

n_neighbors = 15

# import some data to play with
iris = datasets.load_iris()

data = iris.data
target = iris.target

for x, y in zip(data, target):
    print(x)
    print(y)
    break

x = np.array([5.4, 3.3, 7.1, 1.1])

np.linalg.norm(data-x, axis=1).argsort()[:3]

for x_l in data:
    np.linalg.norm(data - x_l, axis=1)

np.linalg.norm(data-x, axis=1)

np.unique(target)
np.array(np.meshgrid(*[np.unique(target) for i in range(len(np.unique(target)))])).T.reshape(-1,3)