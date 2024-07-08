from scipy.optimize import fmin_l_bfgs_b
import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
from tabulate import tabulate
from libs.model_evaluation import DCF, DCF_min, confusion_matrix, bayes_pred_llr
from libs.utils import row, col

def empirical_prior_logodds(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    prior_emp = (X[:, y==1].shape[1]) / X.shape[1]
    return np.log(prior_emp / (1 - prior_emp))


def quadratic_expansion(data):
    data_exp = []

    for i in range(data.shape[1]):
        x = data[:, i:i+1]
        x_exp = np.vstack([col((x @ x.T).ravel()), x])
        data_exp.append(x_exp)

    return np.hstack(data_exp)


class LogisticRegression:
    def __init__(self, lambda_reg: float=0, prior: np.ndarray=None):
        self.lambda_reg = lambda_reg
        if prior != None:
            self.pi = prior
            self.prior_weighted = True
        else:
            self.prior_weighted = False

    def train(self, X: np.ndarray, y, return_opt: bool=False):
        # initialize weights and bias
        x0 = np.zeros(X.shape[0] + 1)
        # optimize using the L-BFGS algorithm
        x_opt, obj_opt, info = fmin_l_bfgs_b(self.logreg_obj, x0, args=[X, y, self.lambda_reg])
        # save parameters 
        w, b = col(x_opt[:-1]), x_opt[-1]
        self.params = (w, b)
        if return_opt:
            return x_opt, obj_opt, info

    def __call__(self, X):
        w, b = self.params
        return ((w.T @ X) + b).ravel()

    def logreg_obj(self, params, X, y, L):
        w, b = params[:-1], params[-1]
        z = (2 * y) - 1
        S = ((col(w).T @ X) + b).ravel()
        if not self.prior_weighted:
            f = (L / 2) * norm(w)**2 + np.mean(np.logaddexp(0, -z * S))
            G = -z / (1 + np.exp(z * S))
            f_grad = np.hstack([((L * w) + np.mean(row(G) * X, 1)).ravel(), np.mean(G)])
        else:
            N1, N0 = X[:, y==1].shape[1], X[:, y==0].shape[1]
            xi = np.where(z==1, self.pi / N1, (1 - self.pi) / N0)
            f = (L / 2) * norm(w)**2 + np.sum(xi * np.logaddexp(0, -z * S))
            G = -z / (1 + np.exp(z * S))
            f_grad = np.hstack([((L * w) + np.sum(xi * row(G) * X, 1)).ravel(), np.sum(xi * G)])
        return f, f_grad
    
    def logreg_obj_mul(self, params, X, y, L):
        D, N, K = X.shape[0], X.shape[1], len(np.unique(y))
        W, b = params[:-K], col(params[-K:])
        W = np.reshape(W, (D, K))
        # compute the matrix of scores for each sample
        S = (W.T @ X) + b
        # numerically stable softmax
        Ylog = S - logsumexp(S, axis=0)
        # one hot encoding
        T = np.zeros((K, N), dtype=np.int32)
        T[y, np.arange(N)] += 1
        f = ((L / 2) * norm(W)**2) - (np.sum(T * Ylog) / N)
        return f

