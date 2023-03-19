# imports
import numpy as np

class Pruner:
    def __init__(self, tree, desired_num_leaves):
        self.tree = tree
        self.desired_num_leaves = desired_num_leaves
        self.num_leaves = 0

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
      twigs = self.catalog_twigs()
      if len(twigs) == 0:
        return tree

      # Calculate the impurity gain of each twig
      impurity_gains = [twig.impurity_gain for twig in twigs]
      # Find the index of the twig with the least impurity gain
      min_index = np.argmin(impurity_gains)
      # Remove the child node of the twig with the least impurity gain
      twig = twigs[min_index]
      twig.left = None
      twig.right = None

      # Set the node's value to be the average of the labels of the samples in the subtree
      twig.value = np.mean(node.samples.label)






