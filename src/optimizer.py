from abc import ABC, abstractmethod


class BaseOptimizer(ABC):
    def __init__(self):
        self._seed = 0

    def _next_seed(self) -> int:
        seed = self._seed
        self._seed += 1
        return seed

    @abstractmethod
    def setup(self):
        # TODO: setup the acquisition function
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def get_best_result(self):
        pass
