from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
from botorch.models import SingleTaskGP


class BaseGPModel(ABC):

    def __init__(self, mean_module: str, covariance_module: str, **kwargs):
        self.mean_module = mean_module
        self.covariance_module = covariance_module
        self.kwargs = kwargs

    @abstractmethod
    def _fit(self, x: torch.Tensor, y: torch.Tensor):
        """practical realization of fitting the Gaussian process model"""
        pass

    def fit(self, x: torch.Tensor, y: torch.Tensor) -> SingleTaskGP:
        self.model = self._fit(x, y)
        return self.model
