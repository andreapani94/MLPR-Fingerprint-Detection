import numpy as np
from density_estimation import logpdf_gau_nd
from scipy.special import logsumexp
from numpy.linalg import svd
from tabulate import tabulate
from libs.utils import col


def logpdf_gmm(X: np.ndarray, gmm_params: list[tuple], return_jointpdf: bool=False):
    logpdf_XjointG = []

    # compute the component g joint log-densities for every sample
    # by summing the log conditionate densities with the prior for every component g
    for w, mu, C in gmm_params:
        logpdf_XcondG = logpdf_gau_nd(X, mu, C)
        logP_G = np.log(w) 
        logpdf_XjointG.append(logpdf_XcondG + logP_G)
    logpdf_XjointG = np.vstack(logpdf_XjointG)
    # marginalize over the components to obtain the GMM density
    logpdf_GMM_X = logsumexp(logpdf_XjointG, axis=0)

    if return_jointpdf:
        return logpdf_GMM_X, logpdf_XjointG
    else: return logpdf_GMM_X

def bound_cov(C, covbound):
    U, s, _ = svd(C)
    s[s < covbound] = covbound
    C_bounded = U @ (col(s) * U.T)
    return C_bounded


def gmm_estim_EM(X: np.ndarray, gmm_params: list[tuple], covbound: float=None, covtype=None):
    N = X.shape[1]

    # E-step: compute responsibilities (component posterior-probabilities)
    logpdf_GMM_X, logpdf_XjointG = logpdf_gmm(X, gmm_params, return_jointpdf=True)
    resp = np.exp(logpdf_XjointG - logpdf_GMM_X)
    gmm_params_upd = []
    # M-step: Update model parameters
    for g in range(len(gmm_params)):
        #   compute statistics
        Zg = np.sum(resp[g])
        Fg = np.sum(resp[g] * X, 1)
        Sg = ((resp[g] * X) @ X.T)
        # update GMM parameters using statistics
        mu_upd = col(Fg / Zg)
        C_upd = (Sg / Zg) - (mu_upd @ mu_upd.T)
        if covtype == 'diag':
            C_upd = C_upd * np.eye(C_upd.shape[0])
        w_upd = Zg / N
        gmm_params_upd.append((w_upd, mu_upd, C_upd))
    if covtype == 'tied':
        Ctied = 0
        for w, _, C in gmm_params_upd:
            Ctied += w * C
        gmm_params_upd = [(w, mu, Ctied) for w, mu, _ in gmm_params_upd]
    if covbound is not None:
        # constraining the eigenvalues of the covariance matrix
        gmm_params_upd = [(w, mu, bound_cov(C, covbound)) for w, mu, C in gmm_params_upd]
                
    return gmm_params_upd          
     

def gmm_split_LBG(X: np.ndarray, gmm_params: list[tuple], a: float, covbound=None, variant=None):
    N = X.shape[1]
    gmm_params_next = []

    for w, mu, C in gmm_params:
        U, s, _ = svd(C)
        d = U[:, 0:1] * s[0]**0.5 * a
        # compute and add the two new components
        gmm_params_next.append((w / 2, mu - d, C))
        gmm_params_next.append((w / 2, mu + d, C))
    
    return gmm_params_next


def print_gmm_results(results, headers):
    print(tabulate(results, headers, tablefmt='grid', floatfmt='.2f'))
    print()

class GMM():
    def __init__(self, params_init: list[tuple]=[], n_components: int=None, covtype='full'):
        self.params = params_init
        self.G = len(params_init) if None else n_components
        self.covtype = covtype

    def __call__(self, X):
        return logpdf_gmm(X, self.params)

    def _train_EM(self, X, ll_t, covbound=None, return_ll=False, verbose=False):
        ll_curr = np.mean(logpdf_gmm(X, self.params))
        ll_delta = None
        step = 0

        while ll_delta is None or ll_delta > ll_t:
            self.params = gmm_estim_EM(X, self.params, covbound, self.covtype)
            ll_next = np.mean(logpdf_gmm(X, self.params))
            ll_delta = abs(ll_curr - ll_next)
            ll_curr = ll_next
            step += 1
            if verbose:
                print(f'GMM optimization - step {step}: average log-likelihood: {ll_next:.8f}')
        
        if return_ll: return ll_curr

    def train(self, X, conv_threshold: float=1e-6, return_ll=False, verbose=False, LBG_displ: float=0.1, covbound: float=None):
        if not self.params:
            # LBG + EM
            mu = np.mean(X, 1, keepdims=True)
            C = ((X - mu) @ (X - mu).T) / X.shape[1]
            if covbound:
                C = bound_cov(C, covbound)
            self.params = [(1, mu, C)]
            gmm_ll = np.mean(logpdf_gmm(X, self.params))

            while len(self.params) < self.G:
                self.params = gmm_split_LBG(X, self.params, LBG_displ, covbound=covbound)
                gmm_ll = self._train_EM(X, conv_threshold, covbound=covbound, return_ll=return_ll, verbose=verbose)
                

            if return_ll: return gmm_ll
        else:
            # EM
            return self._train_EM(X, conv_threshold, covbound=covbound, return_ll=return_ll, verbose=verbose)
