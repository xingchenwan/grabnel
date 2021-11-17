import torch
from torch.distributions import Normal


def graph_expected_improvement(x_star: list, predictor, xi: float = 0.0, in_fill: str = 'best',
                               augmented=False,
                               bias=None):

    mean, variance = predictor.predict(x_star)
    std = torch.sqrt(variance)
    if in_fill == 'best':
        mu_star = torch.max(predictor.y)
    elif in_fill == 'posterior':
        best_idx = torch.argmax(predictor.y)
        mu_star = predictor.predict(predictor.X[best_idx], full_covariance=False)[0]
    else:
        raise NotImplementedError(f'Unknown in fill criterion {in_fill}.')
    gauss = Normal(torch.zeros(1, device=mean.device), torch.ones(1, device=mean.device))
    u = (mean - mu_star - xi) / std
    ucdf = gauss.cdf(u)
    updf = torch.exp(gauss.log_prob(u))
    ei = std * updf + (mean - mu_star - xi) * ucdf
    if augmented:
        sigma_n = predictor.gp.likelihood.noise.detach()
        ei *= (1. - torch.sqrt(torch.tensor(sigma_n, device=mean.device)) / torch.sqrt(sigma_n + variance))
    if bias is not None:
        ei += bias
    return ei


def graph_ucb(x_star: list, predictor, beta: float = None, bias=None):

    mu, variance = predictor.predict(x_star, full_covariance=False)
    std = torch.sqrt(variance)
    beta = 2. if beta is None else beta
    ucb = mu + beta * std
    if bias is not None:
        ucb += bias
    return ucb


def best_mean(x_star: list, predictor):
    mu, _ = predictor.predict(x_star, full_covariance=False)
    return mu
