import random

from collections import namedtuple, deque

import numpy as np
import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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
        if self.update_val != 0:
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

class PER:
    def __init__(self, capacity=10000, eps=0.0001, alpha=0.6, init_beta=0.3, max_episode=100, device=torch.device("cpu")):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.transitions = [None] * capacity
        self.td_error = [None] * capacity
        self.weights = [None] * capacity
        self._len = 0
        self.pointer = 0
        self.eps = eps
        self.alpha = alpha
        self.beta = init_beta
        self.init_beta = init_beta
        self.episode = 0
        self.max_episode = max_episode
        self.device = device
        self.tree_outdated = True
        print('PER')

    def append(self, transition, td_error):
        self.transitions[self.pointer] = transition
        self.td_error[self.pointer] = td_error
        if self._len != self.capacity:
            self._len += 1
        if self.pointer == self.capacity - 1:
            self.pointer = 0
        else:
            self.pointer += 1
        self.tree_outdated = True

    def update_td_error(self, idxs, td_errors):
        for idx, td_error in zip(idxs, td_errors):
            self.td_error[idx] = td_error
        self.tree_outdated = True

    def update_tree(self):
        with torch.no_grad():
            td_tensor = torch.tensor(self.td_error[:self._len], device=self.device)
            priority_tensor = td_tensor.add(self.eps).pow(self.alpha)
            priority_sum = priority_tensor.sum()
            weights_tensor = torch.div(torch.ones(self._len, device=self.device), torch.mul(priority_tensor, self._len)).pow(self.beta)
            final_priority_tensor = priority_tensor.div(priority_sum).mul(weights_tensor)
            for x in range(self._len):
                self.tree.update_leaf(x, final_priority_tensor[x].item())

    def get_random_transition(self):
        idx = self.tree.get_leaf(random.uniform(0, float(self.tree.get_sum()))).idx
        return idx, self.transitions[idx]

    def get_transition_batch(self, batch_size):
        if self.tree_outdated:
            self.update_tree()
            self.tree_outdated = False
        idx_batch = []
        transition_batch = []
        for _ in range(batch_size):
            idx, transition = self.get_random_transition()
            while idx in idx_batch:
                idx, transition = self.get_random_transition()
            idx_batch.append(idx)
            transition_batch.append(transition)
        return idx_batch, transition_batch

    def update_episode(self, episode):
        self.episode = episode
        self.beta = self.init_beta + (1-self.init_beta) * (self.episode/self.max_episode)

    def __len__(self):
        return self._len
