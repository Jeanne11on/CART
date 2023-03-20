# CART
This is an implementation of the Classification and Regression Tree (CART) algorithm from scratch in Python. CART is a decision tree algorithm that can be used for both classification and regression tasks. The goal of the algorithm is to create a binary tree that recursively splits the training data into subsets that are as pure as possible.

Worked on this project: Nan Chen, Lucrezia Certo, Clara Besnard, Yu-hsin LIao, Léopold Granger

## Dependencies
The following dependencies are required:
- NumPy
- Graphviz
- Scikit-learn

## Structure
The project has three files: "CART.py", "pruner.py", and "main.py"
- The "CART.py" file contains the implementation of the decision tree using the Classification and Regression Tree (CART) algorithm.
- The "pruner.py" file contains the implementation of a pruner that can prune the decision tree to prevent overfitting.
- The "main.py" file is the entry point to the program and is used to run the classes on real datasets. It likely includes code that reads in the data, pre-processes it, and trains and evaluates the decision tree and pruner.

## Class Description
### Node class

This is a Python implementation of a decision tree node class, which can be used to construct a decision tree for classification or regression problems. The Node class represents a node in the decision tree and contains the following hyperparameters:

- `feature_index`: the index of the feature to split on.
- `threshold`: the threshold value for the split.
- `value`: the prediction value for the node.
- `left`: the left subtree.
- `right`: the right subtree.
- `impurity_gain`: the impurity gain for the node.

The Node class is a fundamental building block for constructing a decision tree. With this implementation, you can create nodes with specified hyperparameters and use them to construct a decision tree for classification or regression problems.

### CART decision tree class

The CART class is a Python implementation of a decision tree algorithm using the Classification and Regression Trees (CART) approach. The class contains methods to build a decision tree from the input data and output labels, and to make predictions on new data points using the decision tree. The hyperparameters for the decision tree, such as the criterion for splitting, maximum depth, and minimum samples required to split a node, can be specified in the constructor method. The decision tree is built recursively by selecting the best feature to split the data based on the impurity index (Gini or MSE) and creating child nodes for the left and right subsets of data. The class uses a random subspace method to select a random subset of features to consider at each split. The tree-building process stops when a stopping criterion is met, such as reaching the maximum depth or having too few samples to split. The class also includes methods to calculate the impurity and impurity gain for splitting the data, and to split the data into left and right subsets based on a threshold value.


#### Initialization

- `__init__(self, criterion="gini", max_depth=None, min_samples_split=2)`

Initialize the CART object with specified hyperparameters:
| Parameter  | Type | Description |
| ------------- | ------------- | ------------- |
| `criterion`  | str, compulsory (default="gini")  |  The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity in case of classification and "mse" for the Mean Square Error in case of regression.  |
| `max_depth`  | int, optional (default=None)  | The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than `min_samples_split` samples.|
|`min_samples_split` | int (default=2) | The minimum number of samples required to split an internal node. |

#### Prediction

- `fit(self, X, y)`

Build a decision tree from the training set (X, y).

`X` : array-like, shape (n_samples, n_features) - The training input samples.

`y` : array-like, shape (n_samples,) -The target values (class labels) for the training input samples.



- `predict(self, X)`

Predict class or regression value for X. Takes `X` array-like, shape (n_samples, n_features), the input samples. 

Returns `y` : array-like, shape (n_samples,), the predicted class labels or regression values for the input samples.



- `_build_tree(self, X, y, depth=0)`

Build a decision tree recursively.

`X` : array-like, shape (n_samples, n_features), the training input samples.

`y` : array-like, shape (n_samples,), the target values (class labels) for the training input samples.

`depth` : int, optional (default=0), the current depth of the tree.

Returns Object Node, a node of the tree.

<details>
	<summary>Building the tree functions</summary>

- `_get_num_features(self, num_features)`

Returns the number of features to consider at each split.

- `_best_split(self, X, y, feature_indices)`

Find the best feature and threshold to split on based on the impurity gain.
This method finds the best split point for a node in the decision tree based on the information gain of each feature. 

*Parameters:*

`X`: a numpy array of shape (n_samples, n_features) containing the features of the dataset.
`y`: a numpy array of shape (n_samples,) containing the labels of the dataset.
`features_indices`: a list of the indices of the features that are used for splitting the node.

*Returns:*

`split_index`: the index of the feature that yields the highest information gain.
`split_threshold`: the value at which the best feature should be split.
`best_gain`: the information gain of the split.
`best_feature_index`: the index of the feature that yields the highest information gain.

- `_split(column, threshold)`

Splits the data based on the given feature and threshold. It returns the indices of the samples that belong to the left and right sub-nodes, respectively.

*Parameters:*

`column`: a numpy array of shape (n_samples,) containing the feature values of a specific feature.
`threshold`: a scalar representing the value at which the feature should be split.

*Returns:*

`left_indices`: a numpy array of shape (n_left_samples,) containing the indices of the samples that belong to the left sub-node.
`right_indices`: a numpy array of shape (n_right_samples,) containing the indices of the samples that belong to the right sub-node.

- `_impurity_gain(self, y, column, threshold)`

Computes the impurity gain of the split.

*Parameters:*

`y`: a numpy array of shape (n_samples,) containing the labels of the dataset.
`column`: a numpy array of shape (n_samples,) containing the values of the feature to be split.
`threshold`: the threshold value for splitting the node.

*Returns:*
The information gain of the split based on the impurity measure of the parent node and the weighted impurity measures of the child nodes.
	

- `visualize_tree(self, filename)`

Visualizes the decision tree using Graphviz. It creates a graphical representation of the tree and saves it in the specified filename. The function calls two helper functions, `_add_nodes` and `_add_edges`, to add nodes and edges to the graph.

</details>


### Pruner class

The simplest technique is to prune out portions of the tree that result in the least impurity gain. This procedure does not require any additional data, and only bases the pruning on the information that is already computed when the tree is being built from training data.
The process of IG-based pruning requires us to identify “twigs”, nodes whose children are all leaves. “Pruning” a twig removes all of the leaves which are the children of the twig, and makes the twig a leaf. 

The algorithm for pruning is as follows:

1. Catalog all twigs in the tree
2. Count the total number of leaves in the tree.
3. While the number of leaves in the tree exceeds the desired number:
- Find the twig with the least Information Gain
- Remove all child nodes of the twig.
- Relabel twig as a leaf.
- Update the leaf count.
- Update impurity and impurity gain of all nodes
	
	
To this point, we have managed until Update the lead count. Updating the impurity and impurity gain at each node involved in the pruning is to be implemented. In order to to do, we will add a new parameter in the Node initialization: parent, to enable us traverse the tree from the new leave to the root and recompute at each step the impurity. Implementing the functions to update the impurity is yet to be done. Hence, right now, even if the tree is pruned, the performance metric are not correctly updated (there is no change in accuracy or MSE before or after the pruning). 

## Usage

### Classifier decision tree

```python
from decision_tree import CART
from pruner_bis import Pruner

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error


iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42)

cart_classifier = CART(criterion='gini', max_depth=5, min_samples_split=2)
cart_classifier.fit(X_train, y_train)
y_pred = cart_classifier.predict(X_test)
print('Classification Accuracy:', accuracy_score(y_test, y_pred))
cart_classifier.visualize_tree('clf_tree')

# Pruning the tree to 5 leaves
pruner = Pruner(cart_classifier, 5)
pruner.prune()
y_pred_pruned = cart_classifier.predict(X_test)
print('Classification Accuracy:', accuracy_score(y_test, y_pred))
cart_classifier.visualize_tree('clf_tree')
```

### Regression decision tree

```python
# we generate a random dataset
rng = np.random.RandomState(1)
X_reg = np.sort(5 * rng.rand(80, 1), axis=0)
y_reg = np.sin(X_reg).ravel()
y_reg[::5] += 3 * (0.5 - rng.rand(16))
dataset = []
for i in range(len(y_reg)):
  dataset.append([X_reg[i][0], y_reg[i]])
X_train, X_test, y_train, y_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# tree implementation
cart_regressor = CART(criterion='mse', max_depth=5, min_samples_split=2)
cart_regressor.fit(X_train, y_train)
y_pred = cart_regressor.predict(X_test)
print('Regression MSE:', mean_squared_error(y_test, y_pred))
cart_regressor.visualize_tree('reg_tree')


# pruning
pruner = Pruner(cart_regressor, 15)
pruner.prune_tree()
y_pred_pruned = cart_regressor.predict(X_test)
print('Regression MSE:', mean_squared_error(y_test, y_pred))
cart_regressor.visualize_tree('pruned_reg_tree')
```
