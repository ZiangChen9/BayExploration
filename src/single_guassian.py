from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseGP(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.mean = config["mean"]
        self.covariance = config["covariance"]
        self.setup(config=config)

    @abstractmethod
    def setup(self, config: Dict[str, Any]):
        pass

    def get_name(self) -> str:
        return self.__class__.__name__ + str(self.mean) + str(self.covariance)


class SingleTaskGaussianProcess(BaseGP):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        return super().setup(config)
