import random
import time

import numpy as np

class SumTree_new:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity)

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, idx, value):
        # tree index of the leaf node
        tree_idx = idx + self.capacity - 1
        change = value - self.tree[tree_idx]
        self.tree[tree_idx] = value
        self._propagate(tree_idx, change)

    def get_leaf(self, value):
        parent_idx = 0
        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break
            else:
                if value <= self.tree[left_child_idx]:
                    parent_idx = left_child_idx
                else:
                    value -= self.tree[left_child_idx]
                    parent_idx = right_child_idx
        data_idx = leaf_idx - self.capacity + 1
        return data_idx, self.tree[leaf_idx], self.data[data_idx]

    def total(self):
        return self.tree[0]


class Node:
    def __init__(self, left, right, leaf=False, idx=None):
        self.child_left = left
        self.child_right = right
        self.is_leaf = leaf
        self.update_val = 0
        if self.is_leaf:
            self.value = None
        else:
            self.value = self.child_left.value + self.child_right.value
        self.idx = idx
        self.parent = None
        if self.child_left is not None:
            self.child_left.parent = self
        if self.child_right is not None:
            self.child_right.parent = self

    @classmethod
    def create_leaf(cls, idx, value):
        leaf = cls(None, None, leaf=True, idx=idx)
        leaf.value = value
        return leaf

    def get_leaf(self, value):
        if self.is_leaf:
            return self
        else:
            if self.child_left.value >= value:
                return self.child_left.get_leaf(value)
            else:
                return self.child_right.get_leaf(value - self.child_left.value)

    def update(self, value):
        change = value - self.value
        self.value = value
        self.parent.update_val += change

    def propagated_update(self):
        self.value += self.update_val
        if self.parent is not None:
            self.parent.update_val += self.update_val
        self.update_val = 0



class SumTree:
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.leafs = [Node.create_leaf(x, 0) for x in range(capacity)]
        self.nodes = []
        self.outdated = True
        nodes = self.leafs
        leftover = None
        while len(nodes) > 1 or leftover is not None:
            if leftover is not None:
                nodes = leftover + nodes
                leftover = None
            inodes = iter(nodes)
            if len(nodes) % 2 == 1:
                leftover = [next(inodes)]

            zip_nodes = zip(inodes, inodes)
            nodes = [Node(*children) for children in zip_nodes]
            self.nodes.append(nodes)
        self.root = nodes[0]

    def get_leaf(self, value):
        if self.outdated:
            for node_group in self.nodes:
                for node in node_group:
                    node.propagated_update()
            self.outdated = False
        return self.root.get_leaf(value)

    def get_sum(self):
        if self.outdated:
            for node_group in self.nodes:
                for node in node_group:
                    node.propagated_update()
            self.outdated = False
        return self.root.value

    def update_leaf(self, idx, value):
        self.leafs[idx].update(value)
        self.outdated = True




if __name__ == '__main__':
    start_time = time.time()
    tree = SumTree()
    for a, b in zip(range(10000), range(10000, 0, -1)):
        tree.update_leaf(a, b)
    for _ in range(1000):
        test = random.uniform(0, float(tree.get_sum()))
        test1 = tree.get_leaf(test)
    print(time.time() - start_time)

    start_time = time.time()
    tree = SumTree_new()
    for a, b in zip(range(10000), range(10000, 0, -1)):
        tree.update(a, b)
    for _ in range(1000):
        test = random.uniform(0, float(tree.total()))
        test1 = tree.get_leaf(test)
    print(time.time() - start_time)

    start_time = time.time()
    tree = SumTree()
    for a, b in zip(range(10000), range(10000, 0, -1)):
        tree.update_leaf(a, b)
    for _ in range(1000):
        test = random.uniform(0, float(tree.get_sum()))
        test1 = tree.get_leaf(test)
    print(time.time() - start_time)
