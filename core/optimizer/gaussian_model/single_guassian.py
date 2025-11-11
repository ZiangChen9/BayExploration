from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from gpytorch.kernels import MaternKernel, RBFKernel
from gpytorch.means import ConstantMean, LinearMean
from torch import Tensor

from core.optimizer.gaussian_model.base_guassian import BaseGPModel
from core.optimizer.gaussian_model.means import QuadraticMean


MEAN_MODULE_MAP = {
    "constant": ConstantMean,
    "linear": LinearMean,
    "quadratic": QuadraticMean,
}

COVARIANCE_MODULE_MAP = {
    "rbf": RBFKernel,
    "matern1_5": lambda nu: MaternKernel(nu=1.5),
    "matern2_5": lambda nu: MaternKernel(nu=2.5),
}


class ConstantMeanSingleTaskGPModel(BaseGPModel):
    def __init__(self, covariance_module: str, **kwargs):
        super().__init__(
            mean_module="constant", covariance_module=covariance_module, **kwargs
        )

    def _fit(self, x: Tensor, y: Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=MEAN_MODULE_MAP[self.mean_module],
            covar_module=COVARIANCE_MODULE_MAP[self.covariance_module],
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class LinearMeanSingleTaskGPModel(BaseGPModel):
    def __init__(self, covariance_module: str, **kwargs):
        super().__init__(
            mean_module="linear", covariance_module=covariance_module, **kwargs
        )

    def _fit(self, x: Tensor, y: Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=MEAN_MODULE_MAP[self.mean_module],
            covar_module=COVARIANCE_MODULE_MAP[self.covariance_module],
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class QuadraticMeanSingleTaskGPModel(BaseGPModel):
    def __init__(self, covariance_module: str, **kwargs):
        super().__init__(
            mean_module="quadratic", covariance_module=covariance_module, **kwargs
        )

    def _fit(self, x: Tensor, y: Tensor) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=MEAN_MODULE_MAP[self.mean_module],
            covar_module=COVARIANCE_MODULE_MAP[self.covariance_module],
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp


class RBFConstantMeanSingleTaskGPModel(ConstantMeanSingleTaskGPModel):
    def __init__(self, **kwargs):
        super().__init__(covariance_module="rbf")


class Matern1_5ConstantMeanSingleTaskGPModel(ConstantMeanSingleTaskGPModel):
    def __init__(self, **kwargs):
        super().__init__(covariance_module="matern1_5")


class Matern2_5ConstantMeanSingleTaskGPModel(ConstantMeanSingleTaskGPModel):
    def __init__(self, **kwargs):
        super().__init__(covariance_module="matern2_5")
