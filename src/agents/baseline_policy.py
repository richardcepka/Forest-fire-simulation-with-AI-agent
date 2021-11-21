import numpy as np


class BaselinePolicy:

    def __init__(self, output_dim=5):
        self._output_dim = output_dim

    def activate(self, _):
        logits = [0 for _ in range(self._output_dim)]
        best_index = np.random.choice(list(range(self._output_dim))).tolist()
        logits[best_index] = 1
        return logits
