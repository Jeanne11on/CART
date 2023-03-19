# imports
import pandas as pd
import numpy as np
import random
from random import seed
from random import randrange
from math import sqrt

from decision_tree import CART
from pruner_bis import Pruner


## Testing
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


##### Classifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

cart_classifier = CART(criterion='gini', max_depth=5, min_samples_split=2)
cart_classifier.fit(X_train, y_train)
y_pred = cart_classifier.predict(X_test)
print('Classification Accuracy:', accuracy_score(y_test, y_pred))
cart_classifier.visualize_tree('clf_tree')


## We prune the tree
pruner = Pruner(cart_classifier, 5)
pruner.prune_tree()
y_pred_pruned = cart_classifier.predict(X_test)
print('Classification Accuracy:', accuracy_score(y_test, y_pred))
cart_classifier.visualize_tree('pruned_clf_tree')


##### Regressor
rng = np.random.RandomState(1)
X_reg = np.sort(5 * rng.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel()
y_reg[::5] += 3 * (0.5 - rng.rand(16))
dataset = []
for i in range(len(y_reg)):
  dataset.append([X_reg[i][0], y_reg[i]])

X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)


cart_regressor = CART(criterion='mse', max_depth=5, min_samples_split=2)
cart_regressor.fit(X_train, y_train)
y_pred = cart_regressor.predict(X_test)
print('Regression MSE:', mean_squared_error(y_test, y_pred))
cart_regressor.visualize_tree('reg_tree')

pruner = Pruner(cart_regressor, 15)
pruner.prune_tree()
y_pred_pruned = cart_regressor.predict(X_test)
print('Regression MSE:', mean_squared_error(y_test, y_pred))
cart_regressor.visualize_tree('pruned_reg_tree')
