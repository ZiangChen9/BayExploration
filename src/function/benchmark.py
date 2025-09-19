from abc import ABC, abstractmethod
from typing import List


class BaseTestFunction(ABC):
    def __init__(self):
        self.noise_level: float = 0
        self.dim: int = 0
        self.optimum: List[float] = None

    @abstractmethod
    def evaluate(self, x) -> float:
        pass
