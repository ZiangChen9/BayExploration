"""
采集函数基类
"""

from abc import ABC, abstractmethod

from botorch.models import SingleTaskGP


# acf=qMaxValueEntropy(model=gp_1,candidate_set=candidate_set)
class BaseOptimizer(ABC):
    def __init__(self):
        self._seed = 0
        self.acquisition_function = None

    def _next_seed(self) -> int:
        seed = self._seed
        self._seed += 1
        return seed

    @abstractmethod
    def setup(self, pg: SingleTaskGP, config):
        # TODO: 设置好优化器，应接收一个高斯代理模型和采集函数的相关配置
        pass

    @abstractmethod
    def run(self):
        pass

    def get_best_result(self):
        pass
