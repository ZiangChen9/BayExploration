"""
实验层和目标函数和优化器耦合?
一次实验是指一个函数使用一个[或多个采集函数]执行n次迭代，再将这个过程进行m次重复
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import yaml
from botorch import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from gpytorch import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from core.benchmark.base_function import BaseTestFunction


class BaseExperimentFactory(ABC):
    gp_history: Dict[int, Any] = {}  #
    x_history: Dict[int, float] = {}
    y_noised_history: Dict[int, float] = {}
    y_real_history: Dict[int, float] = {}


class ExperimentFactory(BaseExperimentFactory):

    def __init__(self, config: Dict[str, Any]):
        self.init_sample_x = None
        self.test_function = None
        self.config: Dict[str, Any] = config

    @classmethod
    def load_config(cls, config: str | Path | Dict) -> ExperimentFactory:
        # TODO: 若config是字典则直接使用
        suffix = Path(config).suffix.lower()
        with open(config, "r") as f:
            if suffix in [".yaml", ".yml"]:
                config = yaml.safe_load(f)
            elif suffix == ".json":
                config = json.load(f)
        return cls(config)

    @feild_validator("config")
    @classmethod
    def config_val(cls):
        """
        检查配置是否合法
        Returns:
        """
        pass

    def _execute_single_experiment(self, func: BaseTestFunction) -> None:
        # TODO: 实现单次实验的执行 [后期改为多进程可并行]
        # 测试函数与贝叶斯优化器解耦合，测试函数的配置覆盖优化器配置
        self.update_config()
        # 从 test function 处获得函数的维度，最小值，边界等信息
        self.test_function = func
        # 初始函数采样
        _ = self._init_sampling()
        # 拟合高斯

        # 采集函数
        pass

    def _init_sampling(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        initial sampling with sobol engine
        """
        sobol_engine = SobolEngine(dimension=self.test_function.dim, scramble=True)
        draw_number = self.config["init_scale"] * self.test_function.dim
        init_sample_x = sobol_engine.draw(draw_number).to(
            dtype=getattr(torch, self.config["numerical_precision"]),
            device=torch.device(self.config["device"]),
        )
        for i in range(self.test_function.dim):
            self.init_sample_x[:, i] = (
                init_sample_x[:, i]
                * (self.test_function.bounds[i][1] - self.test_function.bounds[i][0])
                + self.test_function.bounds[i][0]
            )
        self.init_sample_y = self.test_function(
            x=self.init_sample_x, noise_level=float(self.config["noise_level"])
        )
        self.init_real_y = self.test_function(x=init_sample_x, noise_level=0)
        return self.init_sample_x, self.init_sample_y, self.init_real_y

    def _fit_gaussian_process(self) -> None:
        self.gp = SingleTaskGP(
            train_X=self.init_sample_x,
            train_Y=self.init_sample_y,
            input_transform=self.config["Normalize"](d=self.test_function.dim),
            outcome_transform=self.config["Standardize"](m=1),
            mean_module=self.config["gaussian"]["mean_fn"],
            covar_module=self.config["gaussian"]["kernel_fn"],
        )
        # Fit
        fit_gpytorch_mll(ExactMarginalLogLikelihood(self.gp.likelihood, self.gp))

    def _update_history(self):
        pass

    @abstractmethod
    def _get_next_point(self):
        pass

    @abstractmethod
    def run(self):
        pass

    def update_config(self):
        pass

    def dump_config(self):
        pass
