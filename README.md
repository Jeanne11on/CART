# CART
This is an implementation of the Classification and Regression Tree (CART) algorithm from scratch in Python. CART is a decision tree algorithm that can be used for both classification and regression tasks. The goal of the algorithm is to create a binary tree that recursively splits the training data into subsets that are as pure as possible.

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

## Usage
### CART decision tree class

- `__init__(self, criterion="gini", max_depth=None, min_samples_split=2)`

Initialize the CART object with specified hyperparameters:
| Parameter  | Type | Description |
| ------------- | ------------- | ------------- |
| `criterion`  | str, compulsory (default="gini")  |  The function to measure the quality of a split. Supported criteria are "gini" for the Gini impurity in case of classification and "mse" for the Mean Square Error in case of regression.  |
| `max_depth`  | int, optional (default=None)  | The maximum depth of the tree. If None, then nodes are expanded until all the leaves contain less than `min_samples_split` samples.|
|`min_samples_split` | int (default=2) | The minimum number of samples required to split an internal node. |

- `fit(self, X, y)`
Build a decision tree from the training set (X, y).

