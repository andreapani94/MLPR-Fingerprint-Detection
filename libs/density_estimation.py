from libs.utils import col
import numpy as np


def logpdf_gau_nd_x(x, mu, C):
    M = x.shape[0]
    _, logdet = np.linalg.slogdet(C)
    C_inv = np.linalg.inv(C)
    y = (-M/2) * np.log(2*np.pi) - 1/2 * logdet - 1/2 * ((x-mu).T  @  C_inv @ (x-mu))
    return y.ravel()

def logpdf_gau_nd(X, mu, C):
    M = X.shape[0]

    _, logdet = np.linalg.slogdet(C)
    C_inv = np.linalg.inv(C)
    const = -(M/2) * np.log(2*np.pi) - 1/2 * logdet
    Y = const -1/2 * np.sum((X - mu) * (C_inv @ (X - mu)), 0)

    return Y

def loglikelihood(X, mu, C):
    return np.sum(logpdf_gau_nd(X, mu, C))

def gau_nd_ml_estimates(data):
    mu_ML = col(np.mean(data))
    C_ML = ((data - mu_ML) @ (data - mu_ML).T) / data.shape[1]

    return mu_ML, C_ML

