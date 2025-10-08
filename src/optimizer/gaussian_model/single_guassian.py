from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from torch import Tensor

from src.optimizer.gaussian_model.base_guassian import BaseGPModel


class SingleTaskGPModel(BaseGPModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _fit(self, x: Tensor, y: Tensor, index: int) -> SingleTaskGP:
        gp = SingleTaskGP(
            train_X=x,
            train_Y=y,
            mean_module=self.mean_module,
            covar_module=self.covariance_module,
        )
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        return gp
