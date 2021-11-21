import torch
import numpy as np

from typing import Tuple


class ForestFireEnviroment:
    EMPTY: int = 0
    TREE: int = 1
    FIRE: int = 2

    def __init__(self, forest_size: Tuple[int], p_empty_tree: float, p_tree_fire: float):
        self._height, self._width = forest_size
        self._p_empty_tree = p_empty_tree
        self._p_tree_fire = p_tree_fire
        self._init_forest_state()

    def _init_forest_state(self):
        self._forest_state = torch.zeros([1, self._height, self._width])

    def update_enviroment(self):
        neighbourhoods = self._make_neighbourhoods()
        self._forest_state = torch.tensor([self._neighbourhood_to_state(neighbourhood)
                                          for neighbourhood in
                                          torch.tensor_split(neighbourhoods, neighbourhoods.shape[0])]).reshape(-1, self._height, self._width)

    @property
    def forest_state(self):
        return self._forest_state

    @forest_state.setter
    def forest_state(self, forest_state):
        self._forest_state = forest_state

    @property
    def num_of_trees(self):
        return (self._forest_state == self.TREE).sum(axis=(1, 2)).tolist()

    @property
    def forest_size(self):
        return (self._height, self._width)

    def _make_neighbourhoods(self):
        pad_forest_state = torch.nn.functional.pad(
            self._forest_state, (1, 1, 1, 1), mode='constant', value=-1)

        kernel_h, kernel_w = 3, 3
        stride = 1
        neighbourhoods = pad_forest_state.unfold(1, kernel_h, stride).unfold(
            2, kernel_w, stride).reshape(-1, kernel_h, kernel_w)
        return neighbourhoods

    def _neighbourhood_to_state(self, neighbourhood):
        neighbourhood = neighbourhood[0, :, :]
        if neighbourhood[1, 1] == self.FIRE:
            return self.EMPTY

        elif neighbourhood[1, 1] == self.TREE:
            if torch.tensor(self.FIRE) in [neighbourhood[0, 1],
                                           neighbourhood[1, 0],
                                           neighbourhood[1, 0],
                                           neighbourhood[2, 1],
                                           neighbourhood[1, 2]]:
                return self.FIRE
            else:
                return np.random.choice([self.TREE, self.FIRE],  p=(1-self._p_tree_fire, self._p_tree_fire))

        elif neighbourhood[1, 1] == self.EMPTY:
            return np.random.choice([self.EMPTY, self.TREE],  p=(1-self._p_empty_tree, self._p_empty_tree))
