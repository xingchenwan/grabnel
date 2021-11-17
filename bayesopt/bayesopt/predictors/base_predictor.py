import torch
from abc import abstractmethod
from bayesopt.bayesopt.acquisitions import graph_expected_improvement, graph_ucb, best_mean
import dgl


class BasePredictor:
    def __init__(self, h: int = 1):
        """
        The base class for predictors based on WL feature extractor.
        :param h:  int. Number of Weisfeiler-Lehman Iterations
        """
        self.h = h
        # Save history for the input and targets for the current predictor class
        self.X, self.y = None, None
        # the lower bound and upper bound for feature vector X and the mean and std of target vector Y
        # will be initialised when the GPWL model is fitted to
        #   some data.
        self.lb, self.ub = None, None
        self.ymean, self.ystd = None, None

    @abstractmethod
    def fit(self, x_train: list, y_train: torch.Tensor):
        """
        Train the predictor on x_train and y_train. note that fit overwrites any data already fit to the predictor
        :param x_train: a list of dgl graphs
        :param y_train: torch.Tensor representing the training targets
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, x_update: list, y_update: torch.Tensor):
        """
        Similar to fit, but append the x_update and y_update to the existing train data (if any, if there
        is no training data, this simply performs fit)
        :param x_update: a list of dgl graphs
        :param y_update: torch.Tensor representing the training targets
        :return: None
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, x_eval: list, **kwargs) -> (torch.Tensor, torch.Tensor):
        """
        Predict the graphs in x_eval using the predictor
        :param x_eval: list of dgl graphs. The test graphs on which we predict
        :param kwargs:
        :return: (mean, variance) torch.Tensor of the same shape of x_eval
        """
        raise NotImplementedError

    def acquisition(self, x_eval: list, acq_func='ei', bias=None, **kwargs) -> torch.Tensor:
        """
        Computes the acquisition function value at the test graphs at x_eval.
        :param x_eval: list of dgl graphs.
        :param acq_func: (str) The acquisition function to be used. Currently supports 'ei' and 'ucb'
        :param bias: any constant, a-priori offset to be added to the acquisition function value of each graph
            in x_eval. If specified, this must be a float or a tensor of the same length as x_eval
        :param kwargs: any additional keyword arguments to passed to the acquisition functions
        :return: acquisition function value evaluated at each graph input of x_eval
        """
        assert acq_func in ['ei', 'ucb', 'mean'], f'Unknown acq function choice {acq_func}'
        if acq_func == 'ei':
            return graph_expected_improvement(x_eval, self, bias=bias, **kwargs)
        elif acq_func == 'ucb':
            return graph_ucb(x_eval, self, bias=bias, **kwargs)
        elif acq_func == 'mean':
            return best_mean(x_eval, self)