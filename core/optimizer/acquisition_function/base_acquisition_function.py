"""采集函数基类"""

from abc import ABC, abstractmethod

from botorch.models import SingleTaskGP
from core.optimizer.gaussian_model.single_guassian import BaseGPModel


class BaseAcquisitionFunction(ABC):
    def __init__(self):
        self._seed = 0
        self.acquisition_function = None

    def _next_seed(self) -> int:
        seed = self._seed
        self._seed += 1
        return seed

    @abstractmethod
    def _setup_optimizer(self, pg: SingleTaskGP, **kwargs):
        pass

    def setup_acquisition_function(self, pg: BaseGPModel, **kwargs):
        self.acquisition_function = self._setup_optimizer(pg=pg.model, **kwargs)
        return self.acquisition_function
