"""
Author: 陈子昂 16179790+ziangchen9@user.noreply.gitee.com
Date: 2025-09-19 16:25:44
LastEditors: 陈子昂 16179790+ziangchen9@user.noreply.gitee.com
LastEditTime: 2025-09-19 16:26:07
FilePath: /BayExploration/src/bayes_optimizer_base.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
"""


class BaseBayesOptimizer:
    def __init__(self, objective_function, bounds, n_iter=10, n_initial_points=5):
        self.objective_function = objective_function
        self.bounds = bounds
        self.n_iter = n_iter
        self.n_initial_points = n_initial_points

    def optimize(self):
        raise NotImplementedError("Subclass must implement this method")

    def _sample_initial_points(self):
        raise NotImplementedError("Subclass must implement this method")

    def _update_model(self, X, y):
        raise NotImplementedError("Subclass must implement this method")

    def _get_next_point(self):
        pass
