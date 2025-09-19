from .base_gaussian import BaseGP


class SingleTaskGaussianProcess(BaseGP):
    def __init__(self):
        super().__init__()

    def setup(self, config):
        return super().setup(config)
