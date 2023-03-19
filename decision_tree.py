# imports
import numpy as np
from graphviz import Digraph

# Create class node
class Node:
    def __init__(self, feature_index=None, threshold=None, value=None, left=None, right=None, impurity_gain=None):
        """
        Initializes a node with some specified hyperparameters.
        """
        self.feature_index = feature_index  # index of the feature to split on
        self.threshold = threshold  # threshold value for the split
        self.value = value  # prediction value for the node
        self.left = left  # left subtree
        self.right = right  # right subtree
        self.impurity_gain = impurity_gain  # impurity gain

class CART:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2):
        """
        Initializes the decision tree with the specified hyperparameters, such as criterion, max_depth, and min_samples_split.
        """
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Fits the decision tree to the input data (X) and output labels (y).
        """
        self.tree = self._build_tree(X, y)


    def predict(self, X):
        """
        Predicts the output label for the input data (X) based on the decision tree.
        """
        return np.array([self._predict(x, self.tree) for x in X])

    def _build_tree(self, X, y, depth=0):
        """
        _build_tree: Builds the decision tree recursively. The first thing the method does is to check if the stopping criteria for tree building are met. 
        If they are, the method returns a leaf node with the predicted value for the subset of data points passed in. If not, the method selects the best
        feature to split the data based on the Gini impurity index. Once the best feature is selected, the data is split into two subsets based on the feature's value. 
        The function then recursively calls itself on each subset of data points until it reaches the stopping criteria or there are no more features to split on.
        The depth parameter tracks the current depth of the tree and is used to enforce the stopping criteria. If the maximum depth is reached or the number of data
        points in a subset is less than or equal to the minimum number of samples required to split a node, a leaf node is returned with the predicted value for the subset of data points.
        The _build_tree method returns a node object that represents a decision in the decision tree. This object contains information about the feature used to split the data, 
        the threshold value for the split, and the left and right child nodes. The left child node contains the subset of data points with values less than or equal to the threshold value,
        while the right child node contains the subset of data points with values greater than the threshold value.
        """
        num_samples, num_features = X.shape
        num_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or num_samples < self.min_samples_split or num_classes == 1:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Find the best split
        feature_indices = np.random.choice(num_features, self._get_num_features(num_features))
        best_feature, best_threshold, gain = self._best_split(X, y, feature_indices)

        # Split the data
        left_indices, right_indices = self._split(X[:, best_feature], best_threshold)
        left_tree = self._build_tree(X[left_indices, :], y[left_indices], depth+1)
        right_tree = self._build_tree(X[right_indices, :], y[right_indices], depth+1)

        return Node(best_feature, best_threshold, left=left_tree, right=right_tree, impurity_gain=gain)

    def _get_num_features(self, num_features):
        """
        _get_num_features: Returns the number of features to consider at each split.
        """
        return int(np.sqrt(num_features))

    def _best_split(self, X, y, feature_indices):
        """
        _best_split: Finds the best feature and threshold to split on based on the impurity gain. Inside the method, 
        it first initializes the best_split variable to None and the lowest_gini variable to inf. 
        It then loops through each potential split in potential_splits, and for each split, it loops through each feature value in the split. 
        For each feature value, it splits the data into two groups based on the feature value: one group where the feature value is less than or equal to the threshold,
        and one group where the feature value is greater than the threshold.
        
        For each split, it calculates the Gini impurity of the two groups using the _calculate_gini method. It then calculates the weighted average of the Gini impurities for the two groups,
        weighted by the size of each group. This is the Gini impurity for the split. If the Gini impurity for the split is less than lowest_gini, it updates lowest_gini to be the Gini impurity
        for the split, and best_split to be the split.
        """
        best_gain = -float('inf')
        split_index, split_threshold = None, None

        for i in feature_indices:
            column = X[:, i]
            thresholds = np.unique(column)
            for threshold in thresholds:
                gain = self._impurity_gain(y, column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_index = i
                    split_threshold = threshold

        return split_index, split_threshold, best_gain

    def _split(self, column, threshold):
        """
        _split: Splits the data based on the feature and threshold.
        """
        left_indices = np.argwhere(column <= threshold).flatten()
        right_indices = np.argwhere(column > threshold).flatten()
        return left_indices, right_indices

    def _impurity_gain(self, y, column, threshold):
        """
        _impurity_gain: Computes the impurity gain of the split.
        """
        parent_impurity = self._impurity(y)
        left_indices, right_indices = self._split(column, threshold)

        if len(left_indices) == 0 or len(right_indices) == 0:
            return 0

        num_left, num_right = len(left_indices), len(right_indices)
        impurity_left = self._impurity(y[left_indices])
        impurity_right = self._impurity(y[right_indices])
        child_impurity = (num_left/len(y)) * impurity_left + (num_right / len(y)) * impurity_right
        return parent_impurity - child_impurity

    def _impurity(self, y):
        """
        _impurity: Computes the impurity of the node based on the criterion (gini or mse).
        """
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'mse':
            return self._mse(y)

    def _gini_impurity(self, y):
        """
        _gini_impurity: Computes the Gini impurity of the node.
        """
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        impurity = 1 - sum(probabilities**2)
        return impurity

    def _mse(self, y):
        """
        _mse: Computes the mean squared error of the node.
        """
        return np.mean((y - np.mean(y))**2)

    def _leaf_value(self, y):
        """
        _leaf_value: Computes the value to assign to a leaf node based on the criterion (most common label or mean).
        """
        if self.criterion == 'gini':
            return self._most_common_label(y)
        elif self.criterion == 'mse':
            return np.mean(y)

    def _most_common_label(self, y):
        """
        _most_common_label: Computes the most common label in the node.
        """
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict(self, x, tree):
        """
        _predict: Predicts the output label for a single input based on the decision tree.
        """
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_index]
        if feature_value <= tree.threshold:
            return self._predict(x, tree.left)
        else:
            return self._predict(x, tree.right)