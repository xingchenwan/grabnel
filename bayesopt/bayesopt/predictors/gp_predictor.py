import gpytorch
import torch
from bayesopt.bayesopt.wl_extractor import WeisfeilerLehmanExtractor
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.constraints.constraints import Interval
import numpy as np
from copy import deepcopy
from bayesopt.bayesopt.utils import to_unit_cube, from_unit_normal, to_unit_normal
from .base_predictor import BasePredictor


class GP(gpytorch.models.ExactGP):
    """Implementation of the exact GP in the gpytorch framework"""
    def __init__(self, train_x, train_y, kernel: gpytorch.kernels.Kernel, likelihood):
        super(GP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class OptimalAssignment(gpytorch.kernels.Kernel):
    """Implementation of the optimal assignment kernel as histogram intersection between two vectors"""
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        """x1 shape = [N x d], x2_shape = [M x d]. This computes the pairwise histogram intersection between the two"""
        ker = torch.zeros(x1.shape[0], x2.shape[0])
        for n in range(x1.shape[0]):
            for m in range(x2.shape[0]):
                ker[n, m] = torch.sum(torch.minimum(x1[n, :].reshape(-1, 1), x2[m, :].reshape(-1, 1)))
        if diag:
            return torch.diag(ker)
        return ker


def train_gp(train_x, train_y, training_iter=50, kernel='linear', verbose=False, init_noise_var=None, hypers={}):
    """Train a GP model. Since in our case we do not have lengthscale, the optimisation is about finding the optimal
    noise only.
    train_x, train_y: the training input/targets (in torch.Tensors) for the GP
    training_iter: the number of optimiser iterations. Set to 0 if you do not wish to optimise
    kernel: 'linear' for the original WL kernel. 'oa' for the optimal assignment variant
    verbose: if True, display diagnostic information during optimisation
    init_noise_var: Initial noise variance. Supply a value here and set training_iter=0 when you have a good knowledge
        of the noise variance a-priori to skip inferring noise from the data.
    hypers: Optional dict of hyperparameters for the GP.
    Return: a trained GP object.
    """

    assert train_x.ndim == 2
    assert train_y.ndim == 1
    assert train_x.shape[0] == train_y.shape[0]

    noise_constraint = Interval(1e-6, 0.1)
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(device=train_x.device, dtype=train_y.dtype)
    if kernel == 'linear':
        k = gpytorch.kernels.LinearKernel()
    elif kernel == 'oa':
        k = OptimalAssignment()
    elif kernel == 'rbf':
        k = gpytorch.kernels.RBFKernel()
    else:
        raise NotImplementedError
    # model
    model = GP(train_x, train_y, k, likelihood).to(device=train_x.device, dtype=train_x.dtype)

    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam([
        {'params': model.parameters()},  # Includes GaussianLikelihood parameters
    ], lr=0.1)

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if hypers:
        model.load_state_dict(hypers)
    else:
        hypers = {}
        hypers["covar_module.outputscale"] = 1.0
        if model.covar_module.has_lengthscale:
            hypers["covar_module.base_kernel.lengthscale"] = np.sqrt(0.01 * 0.5)
        hypers["likelihood.noise"] = 0.005 if init_noise_var is None else init_noise_var
        model.initialize(**hypers)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f. Noise %.3f' % (
                i + 1, training_iter, loss.item(),  model.likelihood.noise.item()
            ))
        optimizer.step()

    model.eval()
    likelihood.eval()
    return model


class GPWL(BasePredictor):
    def __init__(self, kernel: str = 'linear', h: int = 1, noise_var: float = None,
                 extractor_mode: str = 'categorical',
                 node_attr: str = 'node_attr1',):
        """
        A simple GPWL interface which uses GP with WL kernel (note that when the original WL kernel is used, due to
        the linear kernel formulation it is simply WL kernel + Bayesian linear regression

        Note: for linear kernel, it is actually preferable to use bayesian linear regression explicitly
        (bayes_linregress_predictor.py) which saves some overhead from invoking gpytorch, and there seems to be
        problems with gpytorch handling very high-dimensional vectors associated with very large graphs.

        :param kernel: "linear" or "oa" (optimal assigment)
        :param h: maximum number of weisfeiler-lehman iterations
        :param noise_var: the noise variance known a-priori. If None, the noise_variance will be inferred from the
        data via maximum likelihood estimation.
        :param extractor_mode: See extractor documentation
        """
        super().__init__(h)
        self.extractor = WeisfeilerLehmanExtractor(h=h, mode=extractor_mode, node_attr=node_attr)
        self.kernel = kernel
        self.noise_var = noise_var
        self.gp = None

    def fit(self, x_train: list, y_train: torch.Tensor):
        """See BasePredictor"""
        if len(y_train.shape) == 0:  # y_train is a scalar
            y_train = y_train.reshape(1)
        assert len(x_train) == y_train.shape[0]
        assert y_train.ndim == 1
        # Fit the feature extractor with the graph input
        self.extractor.fit(x_train)
        self.X = deepcopy(x_train)
        self.y = deepcopy(y_train)
        # Get the vector representation out
        x_feat_vector = torch.tensor(self.extractor.get_train_features(), dtype=torch.float32)
        # the noise variance is provided, no need for training

        # standardise x_feat_vector into unit hypercube [0, 1]^d
        self.lb, self.ub = torch.min(x_feat_vector, dim=0)[0]+1e-3, torch.max(x_feat_vector, dim=0)[0]-1e-3
        x_feat_vector_gp = to_unit_cube(x_feat_vector, self.lb, self.ub)
        # normalise y vector into unit normal distribution
        self.ymean, self.ystd = torch.mean(y_train), torch.std(y_train)
        y_train_normal = to_unit_normal(y_train, self.ymean, self.ystd)
        if self.noise_var is not None:
            self.gp = train_gp(x_feat_vector_gp, y_train_normal, training_iter=0, kernel=self.kernel, init_noise_var=self.noise_var)
        else:
            self.gp = train_gp(x_feat_vector_gp, y_train_normal, kernel=self.kernel)

    def update(self, x_update: list, y_update: torch.Tensor):
        """See BasePredictor"""
        if len(y_update.shape) == 0:  # y_train is a scalar
            y_update = y_update.reshape(1)
        assert len(x_update) == y_update.shape[0]
        assert y_update.ndim == 1
        self.extractor.update(x_update)
        x_feat_vector = torch.tensor(self.extractor.get_train_features(), dtype=torch.float32)

        # update the lb and ub, in case new information changes those
        self.lb, self.ub = torch.min(x_feat_vector, dim=0)[0]+1e-3, torch.max(x_feat_vector, dim=0)[0]-1e-3
        x_feat_vector_gp = to_unit_cube(x_feat_vector, self.lb, self.ub)
        self.X += deepcopy(x_update)
        self.y = torch.cat((self.y, y_update))
        self.ymean, self.ystd = torch.mean(self.y), torch.std(self.y)
        y = to_unit_normal(deepcopy(self.y), self.ymean, self.ystd)
        if self.noise_var is not None:
            self.gp = train_gp(x_feat_vector_gp, y, training_iter=0, kernel=self.kernel, init_noise_var=self.noise_var)
        else:
            self.gp = train_gp(x_feat_vector_gp, y, kernel=self.kernel)

    def predict(self, x_eval: list, full_covariance=False, include_noise_variance=False):
        """
        See BasePredict
        :param full_covariance: bool. Whether return the full covariance (shape N x N where N = len(x_eval)
        :param include_noise_variance: bool. Whether include noise variance in the prediction. This does not impact
            the posterior mean inference, but the posterior variance inference will be enlarged accordingly if this flag
            is True
        :return: (mean, variance (if full_covariance=True) or the full covariance matrix (if full_covariance=True))
        """
        if self.gp is None:
            raise ValueError("The GPWL object is not fitted to any data yet! Call fit or update to do so first.")
        x_feat_vector = torch.tensor(self.extractor.transform(x_eval), dtype=torch.float32)
        x_feat_vector = to_unit_cube(x_feat_vector, self.lb, self.ub)
        self.gp.eval()
        pred = self.gp(x_feat_vector)
        # print(pred.mean)
        if include_noise_variance:
            self.gp.likelihood.eval()
            pred = self.gp.likelihood(pred)
        mean, variance = pred.mean.detach(), pred.variance.detach()
        mean = from_unit_normal(mean, self.ymean, self.ystd)
        variance = from_unit_normal(variance, self.ymean, self.ystd, scale_variance=True)
        if full_covariance:
            covar = pred.covariance_matrix.detach()
            covar *= (self.ystd ** 2)
            return mean, covar
        return mean, variance
