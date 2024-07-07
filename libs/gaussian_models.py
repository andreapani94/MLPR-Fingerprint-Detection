import numpy as np
import sklearn.datasets
import scipy.special
from dimensionality_reduction import lda

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def logpdf_gau_nd(X, mu, C):
    M = X.shape[0]
    _, logdet = np.linalg.slogdet(C)
    C_inv = np.linalg.inv(C)
    const = -(M/2) * np.log(2*np.pi) - 1/2 * logdet
    Y = const -1/2 * np.sum((X - mu) * (C_inv @ (X - mu)), 0)
    return Y

class MVG:
    def __init__(self):
        self.params = {}

    def fit(self, X, y):
        """
            Estimate MVG parameters by Maximum likelihood
        """
        self.c = np.unique(y)
        for c_i in self.c:
            D_c = X[:, y==c_i]
            mu_c = col(np.mean(D_c, 1))
            C_c = (1 / D_c.shape[1]) * ((D_c - mu_c) @ (D_c - mu_c).T)
            self.params[c_i] = (mu_c, C_c)

    def __call__(self, X, prior=None, log_domain: bool = True):
        if prior == None:
            # uniform priors
            prior = col(np.full_like(self.c, 1/len(self.c), np.float32))
        if len(self.c) > 2:
            # multiclass
            Sjoint = self._Sjoint(X, prior, log_domain)
            Smarg = self._Smarg(Sjoint, log_domain)
            Spost = self._Spost(Sjoint, Smarg)
            return Spost
        else:
            # binary
            llr = self._llr(X)
            return llr

    def _llr(self, X):
        S = []
        for c_i in self.c:
                mu, C = self.params[c_i]
                S.append(logpdf_gau_nd(X, mu, C))
        S = np.vstack(S)
        llr = S[1] - S[0]
        return llr

    def _Sjoint(self, D, p, log_domain: bool = True):
        S = []
        if log_domain:
            for c_i in self.c:
                mu, C = self.params[c_i]
                S.append(logpdf_gau_nd(D, mu, C))
            S = np.vstack(S)
            Sjoint = S + np.log(p)
        else:
            for c_i in self.c:
                mu, C = self.params[c_i]
                S.append(np.exp(logpdf_gau_nd(D, mu, C))) 
            S = np.vstack(S)
            Sjoint = S * p
        return Sjoint
    
    def _Smarg(self, Sjoint, log_domain: bool = True):
        if log_domain:
            Smarg = scipy.special.logsumexp(Sjoint, 0)
        else:
            Smarg = np.sum(Sjoint, 0)
        return Smarg
    
    def _Spost(self, Sjoint, Smarg, log_domain: bool = True):
        if log_domain:
            Spost = Sjoint - Smarg
        else:
            Spost = Sjoint / Smarg
        return Spost
    
class NaiveBayesMVG(MVG):
    def __init__(self):
        super().__init__()

    def fit(self, D, l):
        """
            Estimate MVG parameters by Maximum likelihood
        """
        super().fit(D, l)
        for c_i in self.c:
            mu, C = self.params[c_i]
            I = np.eye(D.shape[0])
            C *= I
            self.params[c_i] = (mu, C)

class TiedCovarianceMVG(MVG):
    def __init__(self):
        super().__init__()

    def fit(self, D, l):
        """
            Estimate MVG parameters by Maximum likelihood
        """
        super().fit(D, l)
        C_tied = 0
        for c_i in self.c:
            _, C = self.params[c_i]
            Nc = D[:, l==c_i].shape[1]
            C_tied += C * Nc
        C_tied /= D.shape[1]
        for c_i in self.c:
            mu, _ = self.params[c_i]
            self.params[c_i] = (mu, C_tied)
