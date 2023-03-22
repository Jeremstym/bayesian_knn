from sklearn.neighbors import KNeighborsClassifier
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

