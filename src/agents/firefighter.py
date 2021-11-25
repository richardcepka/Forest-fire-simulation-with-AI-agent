import torch
import numpy as np
from neat.nn import FeedForwardNetwork

from typing import Tuple

from agents.policy import Policy


class Firefigheter:
    STAY: int = 0
    LEFT: int = 1
    RIGHT: int = 2
    UP: int = 3
    DOWN: int = 4

    EMPTY: int = 0
    TREE: int = 1
    FIRE: int = 2
    AGENT: int = 3

    def __init__(self, init_position: Tuple[int], forest_size: Tuple[int], policy: Policy):
        self._index_i, self._index_j = init_position
        self._height, self._width = forest_size
        self._policy = policy

        self._saved_trees = 0

    def make_action(self, forest_state: torch.Tensor):
        #forest_state_list = forest_state.flatten().tolist()
        #forest_state_list = list(self.position) + forest_state_list
        #logits = np.array(self._policy.activate(forest_state_list))
        neighbourhood_states = []
        for i, j in [(self._index_i + 1, self._index_j),
                     (self._index_i - 1, self._index_j),
                     (self._index_i, self._index_j),
                     (self._index_i, self._index_j + 1),
                     (self._index_i, self._index_j-1)]:
            try:
                neighbourhood_states.append(int(forest_state[0, i, j]))
            except:
                neighbourhood_states.append(-1000)

        logits = np.array(self._policy.activate(neighbourhood_states))
        propabilities = logits/np.sum(logits)
        action = np.argmax(propabilities)
        if action == self.STAY:
            pass
        elif action == self.LEFT and self._index_j - 1 >= 0:
            self._index_j -= 1
        elif action == self.RIGHT and self._index_j + 1 < self._width:
            self._index_j += 1
        elif action == self.UP and self._index_i - 1 >= 0:
            self._index_i -= 1
        elif action == self.DOWN and self._index_i + 1 < self._height:
            self._index_i += 1

        if int(forest_state[0, self._index_i, self._index_j]) == self.FIRE:
            forest_state[0, self._index_i, self._index_j] = self.TREE
            self._saved_trees += 1

        return forest_state

    @property
    def saved_trees(self):
        return self._saved_trees

    @property
    def position(self):
        return (self._index_i, self._index_j)

    def env_agent(self, forest_state: torch.Tensor):
        forest_state_agent = torch.clone(forest_state)
        forest_state_agent[0, self._index_i, self._index_j] = self.AGENT
        return forest_state_agent
