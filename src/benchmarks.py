from abc import ABC, abstractmethod
from typing import List, Dict, Any

import numpy as np
import torch


class BaseTestFunction(ABC):
    _dim: int = 1
    _optimal: List[float] = []
    _optimal_value: float = 0.0
    _bound: List[List[float, float]] = []
    _default_kwargs = {
        "dtype": torch.float64,
        "device": torch.device("cpu"),
    }

    def __init__(self, noise_level: float = 0.05, **kwargs):
        self.noise_level: float = noise_level
        self.kwargs = {**self._default_kwargs, **kwargs}

    @abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def optimal(self) -> torch.Tensor:
        return torch.tensor(self._optimal, **self.kwargs)

    @property
    def bound(self) -> torch.Tensor:
        return torch.tensor(self._bound, **self.kwargs)

    @abstractmethod
    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Ackley(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = 0.0
    _bound = [[-32.7680, -32.7680], [32.7680, 32.7680]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level, **kwargs)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = (
            torch.exp(-0.2 * torch.sqrt(0.5 * (x1**2 + x2**2)))
            - torch.exp(0.5 * (torch.cos(2 * np.pi * x1) + torch.cos(2 * np.pi * x2)))
            + 1
        )
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)


class Booth(BaseTestFunction):
    _dim = 2
    _optimal = [1.0, 3.0]
    _optimal_value = 0.0
    _bound = [[-10.0, -10.0], [10.0, 10.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level, **kwargs)

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x_1, x_2 = x[..., 0], x[..., 1]
        y = (x_1 + 2 * x_2 - 7) ** 2 + (2 * x_1 + x_2 - 5) ** 2
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)


class Bohachevsky(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = 0.0
    _bound = [[-100.0, -100.0], [100.0, 100.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level=noise_level)
        self.kwargs = {**self._default_kwargs, **kwargs}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = (
            x1**2
            + 2 * x2**2
            - 0.3 * torch.cos(3 * torch.pi * x1)
            - 0.4 * torch.cos(4 * torch.pi * x2)
            + 0.7
        )
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)


class Easom(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = -1.0
    _bound = [[-5.0, -5.0], [5.0, 5.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level=noise_level)
        self.kwargs = {**self._default_kwargs, **kwargs}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = (
            -torch.cos(x1)
            * torch.cos(x2)
            * torch.exp(-((x1 - torch.pi) ** 2 + (x2 - torch.pi) ** 2))
        )
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)


class ThreeHumpCamel(BaseTestFunction):
    _dim = 2
    _optimal = [0.0, 0.0]
    _optimal_value = 0.0
    _bound = [[-1.0, -1.0], [1.0, 1.0]]

    def __init__(self, noise_level: float = 0.05, **kwargs):
        super().__init__(noise_level=noise_level)
        self.kwargs = {**self._default_kwargs, **kwargs}

    def evaluate(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x[..., 0], x[..., 1]
        y = 2.0 * x1**2 - 1.05 * x1**4 + x1**6 / 6.0 + x1 * x2 + x2**2
        noise = torch.randn_like(y) * self.noise_level
        return (y + noise).unsqueeze(-1)
