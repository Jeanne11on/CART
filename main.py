# imports
from decision_tree import CART

## Testing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

##### Classifier
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)
cart_classifier = CART(criterion='gini', max_depth=5, min_samples_split=2)
cart_classifier.fit(X_train, y_train)
y_pred = cart_classifier.predict(X_test)
print('Classification Accuracy:', accuracy_score(y_test, y_pred))
cart_classifier.visualize_tree('whole_tree')

'''
#pruner = Pruner()
pruner = Pruner(cart_classifier.tree, 5)
twigs = pruner.catalog_twigs()
nb_leaves = pruner.count_leaves()
twig_1 = twigs[0]
print(twig_1.threshold, twig_1.impurity_gain, twig_1.left.value)
'''

#cart_classifier.fit(X_train, y_train)
#y_pred = cart_classifier.predict(X_test)
#print('Classification Accuracy pruned:', accuracy_score(y_test, y_pred))
#cart_classifier.visualize_tree('pruned_tree')


## We prune the tree
#pruner = Pruner(cart_classifier)
#pruner.prune()
#y_pred_pruned = cart_classifier.predict(X_test)
#print('Classification Accuracy:', accuracy_score(y_test, y_pred))

'''
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
'''




