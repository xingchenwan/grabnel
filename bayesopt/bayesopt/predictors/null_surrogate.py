# null surrogate

from .base_predictor import BasePredictor
import torch
from copy import deepcopy


class NullSurrogate(BasePredictor):
    def __init__(self, h: int = None, ):
        """
        Null surrogate
        :param h: not required or used. For consistency of APi
        """
        super().__init__(h=h)

    def fit(self, x_train: list, y_train: torch.Tensor):
        self.X = deepcopy(x_train)
        self.y = deepcopy(y_train)

    def update(self, x_update: list, y_update: torch.Tensor):
        self.X += deepcopy(x_update)
        self.y = torch.cat((self.y, y_update))

    def predict(self, x_eval: list, **kwargs) -> (torch.Tensor, torch.Tensor):
        mean = torch.zeros(len(x_eval))
        vars = torch.ones(len(x_eval))
        return mean, vars
