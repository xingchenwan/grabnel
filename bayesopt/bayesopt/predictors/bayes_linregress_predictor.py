from .base_predictor import BasePredictor
import torch
from copy import deepcopy
from bayesopt.bayesopt.utils import to_unit_cube, to_unit_normal, from_unit_normal
from bayesopt.bayesopt.wl_extractor import WeisfeilerLehmanExtractor
from sklearn import linear_model
import numpy as np


class BayesianLinearRegression(BasePredictor):

    def __init__(self, h=1, max_step=1000, verbose=False, ard=True,
                 extractor_mode='categorical',
                 node_attr='node_attr1',
                 **linregress_params):
        """
        Bayesian Linear Regression predictor with the WL feature extractor. This function uses the scikit-learn
            implementation of the Bayesian Linear regression internally.
        :param h: maximum number of weisfeiler-lehman iterations
        :param max_step: maximum number of training steps for ELBO optimisation for the linear regressor
        :param verbose: whether to enable verbose mode
        :param ard: whether to use automatic relevance determination regression.
            If True, we use the instance of sklearn.linear.ARDRegression, which relaxes the assumption that the Gaussian
                distribution of parameter weights is spherical, and variance might vary per parameter. Empirically this
                leads to more sparse representation and thus potentially better results.
            If False, we use the sklearn.linear.BayesianRidge regressor
        :param linregress_params: Any keyword parameters to be passed to the sklearn regressor. See Sklearn documentation
            for more information
        :param extractor_mode: See the extractor documentation.
        """
        super().__init__(h=h)
        self.extractor = WeisfeilerLehmanExtractor(h=h, mode=extractor_mode, node_attr=node_attr)
        self.max_step = max_step
        self.verbose = verbose

        self.ard = ard
        # save the BayesLinRegress model
        self.model = None
        # save the feature vector of WL in case of use
        self.X_feat = None
        self.linregress_params = linregress_params

    def fit(self, x_train: list, y_train: torch.Tensor):
        # if len(y_train.shape) == 0:  # y_train is a scalar
        #     y_train = y_train.reshape(1)
        y_train = y_train.reshape(-1)
        assert len(x_train) == y_train.shape[0]
        assert y_train.ndim == 1
        # Fit the feature extractor with the graph input
        self.extractor.fit(x_train)
        self.X = deepcopy(x_train)
        self.y = deepcopy(y_train)
        # Get the vector representation out
        x_feat_vector = self.extractor.get_train_features().astype(np.float32)
        # the noise variance is provided, no need for training

        # remove any rows
        # print(x_feat_vector)
        x_feat_vector = x_feat_vector[~np.isnan(x_feat_vector).any(axis=1)]
        # self.v = x_feat_vector

        # standardise x_feat_vector into unit hypercube [0, 1]^d
        self.lb, self.ub = np.min(x_feat_vector, axis=0)+1e-3, np.max(x_feat_vector, axis=0)-1e-3
        x_feat_vector_gp = to_unit_cube(x_feat_vector, self.lb, self.ub)
        self.X_feat = deepcopy(x_feat_vector_gp)
        # normalise y vector into unit normal distribution
        self.ymean, self.ystd = torch.mean(y_train), torch.std(y_train)
        if self.ystd > 0:
            y_train_normal = to_unit_normal(y_train, self.ymean, self.ystd)
        else:
            y_train_normal = y_train

        if self.ard:  self.model = linear_model.ARDRegression(**self.linregress_params)
        else:  self.model = linear_model.BayesianRidge(**self.linregress_params)
        self.model.fit(x_feat_vector_gp, y_train_normal.numpy())
        # pyro.clear_param_store()
        # self.model = PyroBayesianRegression(x_feat_vector_gp.shape[1], 1)
        # guide = AutoDiagonalNormal(self.model)
        # optimiser = pyro.optim.Adam({'lr': 0.03})
        # svi = SVI(self.model, guide, optimiser, loss=Trace_ELBO())
        #
        # for j in range(self.max_step):
        #     # calculate the loss and take a gradient step
        #     loss = svi.step(x_feat_vector_gp, y_train_normal)
        #     if self.verbose and j % 100 == 0:
        #         print("[iteration %04d] loss: %.4f" % (j + 1, loss / y_train_normal.shape[0]))

    def update(self, x_update: list, y_update: torch.Tensor):
        # if len(y_update.shape) == 0:  # y_train is a scalar
        #     y_update = y_update.reshape(1)
        y_update = y_update.reshape(-1)
        assert len(x_update) == y_update.shape[0]
        assert y_update.ndim == 1
        self.extractor.update(x_update)
        x_feat_vector = self.extractor.get_train_features()
        self.X_feat = deepcopy(x_feat_vector)

        # remove any rows
        # print(x_feat_vector)
        x_feat_vector = x_feat_vector[~np.isnan(x_feat_vector).any(axis=1)]

        # update the lb and ub, in case new information changes those
        self.lb, self.ub = np.min(x_feat_vector, axis=0)+1e-03, np.max(x_feat_vector, axis=0)-1e-3
        x_feat_vector_gp = to_unit_cube(x_feat_vector, self.lb, self.ub)
        self.X += deepcopy(x_update)
        self.y = torch.cat((self.y, y_update))
        self.ymean, self.ystd = torch.mean(self.y), torch.std(self.y)
        if self.ystd > 0:
            y = to_unit_normal(self.y, self.ymean, self.ystd)
        else:
            y = self.y
        # pyro.clear_param_store()
        # self.model = PyroBayesianRegression(x_feat_vector_gp.shape[1], 1)
        # guide = AutoDiagonalNormal(self.model)
        # optimiser = pyro.optim.Adam({'lr': 0.03})
        # svi = SVI(self.model, guide, optimiser, loss=Trace_ELBO())
        #
        # for j in range(self.max_step):
        #     # calculate the loss and take a gradient step
        #     loss = svi.step(x_feat_vector_gp, y)
        #     if self.verbose and j % 100 == 0:
        #         print("[iteration %04d] loss: %.4f" % (j + 1, loss / y.shape[0]))
        if self.ard:
            self.model = linear_model.ARDRegression(**self.linregress_params)
        else:
            self.model = linear_model.BayesianRidge(**self.linregress_params)
        self.model.fit(x_feat_vector_gp, y)

    def predict(self, x_eval: list, include_noise_variance=False, **kwargs) -> (torch.Tensor, torch.Tensor):
        if self.model is None:
            raise ValueError("The GPWL object is not fitted to any data yet! Call fit or update to do so first.")
        x_feat_vector = self.extractor.transform(x_eval)
        x_feat_vector = to_unit_cube(x_feat_vector, self.lb, self.ub)
        mean, std = self.model.predict(x_feat_vector, return_std=True)
        variance = std ** 2

        if not include_noise_variance:
            # alpha_ is the imputed noise precision
            variance -= np.min(variance) - 1e5
            estimate_noise_var = np.mean(self.model.predict(self.X_feat, return_std=True)[1] ** 2)
            # estimate_noise_var = 1. / self.model.alpha_     # alpha_ is the noise precision
            variance -= estimate_noise_var

        mean = from_unit_normal(mean, self.ymean.numpy(), self.ystd.numpy())
        variance = from_unit_normal(variance, self.ymean.numpy(), self.ystd.numpy(), scale_variance=True)
        std = np.sqrt(variance)

        return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)
