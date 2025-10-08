"""
高斯过程应只和目标函数低耦合
"""

import abc
from abc import abstractmethod
from typing import Dict, Any

from botorch.models import SingleTaskGP
from torch import Tensor


class BaseGPModel(abc.ABC):
    # TODO:限定键只能自增,使用字典是为了满足多线程的要求
    # TODO:调整这些记录到experiment层，不要再高斯模型处存储这些记录
    gp_history: Dict[int, Any] = {}
    x_history: Dict[int, Tensor] = {}
    y_history: Dict[int, Tensor] = {}

    def __init__(self, **kwargs):
        self.mean_module = None
        self.covariance_module = None
        self._setup()

    def _setup(self, **kwargs):
        # TODO:设置模型的归一/标准化，以及均值函数和协方差
        pass

    @abstractmethod
    def _fit(self, x: Tensor, y: Tensor, index: int):
        # TODO:拟合高斯过程并记录对应的历史
        pass

    def execute(self, x: Tensor, y: Tensor, index: int) -> SingleTaskGP:
        if self.gp_history is None:
            self._setup()
        gp = self._fit(x, y, index)
        self.gp_history[index] = gp
        self.x_history[index] = x
        self.y_history[index] = y
        return gp
