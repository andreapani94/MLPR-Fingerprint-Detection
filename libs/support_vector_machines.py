import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
from tabulate import tabulate
from libs.utils import row, col


def obj_primal(D: np.ndarray, z: np.ndarray, a: np.ndarray, C: int):
    w_ext = col(np.sum(a * z * D, axis=1))
    S = 1 - (z * (w_ext.T @ D))
    y = (0.5 * norm(w_ext)**2) + (C * np.sum(np.maximum(0, S)))
    return w_ext, y

def obj_dual(a: np.ndarray, D: np.ndarray, z: np.ndarray, kern=None, K=1):
    N: int = D.shape[1]
    a = col(a)
    # fast version
    if kern is None:
        H = row(z) * (D.T @ D) * row(z).T
    else: 
        H = row(z) * (kern(D, D) + K) * row(z).T
    # #assert(np.allclose(H_slow, H))
    # H = H_slow
    ones = col(np.ones(N))
    y = (0.5 * (a.T @ H @ a)) - (a.T @ ones).item()
    ygrad = ((H @ a) - ones).ravel()
    return y, ygrad 

class SVM():
    def __init__(self, C: float, K: float, kernel=None):
        self.C = C
        self.K = K
        self.kernel = kernel

    def __call__(self, X):
        if self.kernel is None:
            w, b = col(self.params[0]), self.params[1]
            S = w.T @ X + b
            return S.ravel()
        else:
            a, z = self.params
            Skernel = self.kernel(self.Xtrain, X) + self.K
            return np.sum(a * z * Skernel, 0).ravel()
        

    def train(self, X, y, return_opt=False, opt_precision: float=1e7):
        N = X.shape[1]
        x0 = np.zeros(N)
        bounds = [(0, self.C) for _ in range(N)]
        # extend the features
        X_ext = np.vstack([X, np.full((1, N), self.K)])
        # encode the labels as 1, -1
        z = np.where(y > 0, 1, -1)
        a_opt, dual_opt, _ = fmin_l_bfgs_b(obj_dual, args=(X_ext, z, self.kernel, self.K), x0=x0, 
                                                bounds=bounds, factr=opt_precision)
        if self.kernel is None:
            # recover the primal solution to obtain w extended
            w_ext, primal_opt = obj_primal(X_ext, z, a_opt, self.C)
            w, b = w_ext[:-1], w_ext[-1] * self.K # rescale b by K (for cases when K is not 1)
            self.params = (w.ravel(), b.ravel())
        else:
            # save the training samples for scoring
            self.Xtrain = X
            self.params = (col(a_opt), col(z))

        if return_opt:
            if self.kernel is None:
                return abs(dual_opt), primal_opt
            else: 
                return abs(dual_opt)
        

def rbf_kernel(g: float):

    def rbf_kernel_parametrized(x1: np.ndarray, x2: np.ndarray):
        x1_norm = col(np.sum(x1**2, 0))
        x2_norm = row(np.sum(x2**2, 0))
        return np.exp(-g * ((x1_norm + x2_norm - 2 * (x1.T @ x2))))
    
    return rbf_kernel_parametrized
    

def poly_kernel(d: float, c: float):

    def poly_kernel_parametrized(x1: np.ndarray, x2: np.ndarray):
        return ((x1.T @ x2) + c)**d
    
    return poly_kernel_parametrized
    

def print_svm_results(data):
    headers = ['K', 'C', 'Primal loss', 'Dual loss', 'Duality gap', 'Error rate', 'DCF', 'minDCF']
    print(tabulate(data, headers, tablefmt="grid"))
    print()

def print_kernsvm_results(data):
    headers = ['K', 'C', 'Kernel', 'Dual loss', 'Error rate', 'DCF', 'minDCF']
    print(tabulate(data, headers, tablefmt="grid"))
    print()
