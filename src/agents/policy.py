import numpy as np
import neat

from abc import ABC, abstractmethod


class Policy(ABC):

    @abstractmethod
    def activate(self, x):
        pass


class RadnomPolicy(Policy):

    def __init__(self, n_actions=5):
        self._n_actions = n_actions

    def activate(self, _):
        logits = [0 for _ in range(self._n_actions)]
        best_index = np.random.choice(list(range(self._n_actions))).tolist()
        logits[best_index] = 1
        return logits


class LazyPolicy(Policy):
    STAY: int = 0

    def __init__(self, n_actions=5):
        self._n_actions = n_actions

    def activate(self, _):
        logits = [0 for _ in range(self._n_actions)]
        logits[self.STAY] = 1
        return logits


class NEATPolicy(Policy):

    def __init__(self, genome, config):
        self._policy = neat.nn.FeedForwardNetwork.create(genome, config)

    def activate(self, x):
        return self._policy.activate(x)
