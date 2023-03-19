# imports
import numpy as np

class Pruner:
    def __init__(self, tree, desired_num_leaves):
        self.tree = tree.tree
        self.desired_num_leaves = desired_num_leaves
        self.num_leaves = 0
        self.criterion = tree.criterion

    def catalog_twigs(self):
        twigs = []
        self._catalog(self.tree, twigs)
        return twigs

    def _catalog(self, node, twigs):
        if node.value is None:
            if node.left.value is not None and node.right.value is not None:
                twigs.append(node)
            self._catalog(node.left, twigs)
            self._catalog(node.right, twigs)

    def count_leaves(self):
        leaves = []
        self._count_leaves(self.tree, leaves)
        return len(leaves)

    def _count_leaves(self, node, leaves):
        if node.value is not None:
            leaves.append(node)
        else:
            self._count_leaves(node.left, leaves)
            self._count_leaves(node.right, leaves)

    def prune_tree(self):
      '''For now, finds the twig with least impurity gain, deletes its children and transforms it into a leaf
      '''
      twigs = self.catalog_twigs()
      if len(twigs) == 0:
        return self.tree

      # we get the number of leaves
      self.num_leaves = self.count_leaves()

      while self.num_leaves > self.desired_num_leaves:
        # Calculate the impurity gain of each twig
        impurity_gains = [twig.impurity_gain for twig in twigs]
        # Find the index of the twig with the least impurity gain
        min_index = np.argmin(impurity_gains)
        # Remove the child node of the twig with the least impurity gain
        twig = twigs[min_index]
        sample = [twig.left.value, twig.right.value]
        twig.left = None
        twig.right = None
        twig.impurity_gain = 0

        # Set the node's value to be the average of the labels of the samples in the subtree
        if self.criterion == 'gini':
            twig.value = np.max(sample)
        elif self.criterion == 'mse':
            twig.value = np.mean(sample)

        # we update twigs based on the new tree
        twigs = self.catalog_twigs()

        # we update count_leaves based on the new tree
        self.num_leaves = self.count_leaves()

        #Todo: update the impurity and impurity gain of the nodes involved in this pruning 






