import numpy as np
import sklearn.datasets
import scipy.special
from dimensionality_reduction import lda

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def split_2to1(D, l, seed=0):
    ntrain = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxtrain = idx[:ntrain]
    idxval = idx[ntrain:]

    Dtr = D[:, idxtrain]
    Dval = D[:, idxval]
    ltr = l[idxtrain]
    lval = l[idxval]

    return (Dtr, ltr), (Dval, lval)

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

    def fit(self, D, l):
        """
            Estimate MVG parameters by Maximum likelihood
        """
        self.c = np.unique(l)
        for c_i in self.c:
            D_c = D[:, l==c_i]
            mu_c = col(np.mean(D_c, 1))
            C_c = (1 / D_c.shape[1]) * ((D_c - mu_c) @ (D_c - mu_c).T)
            self.params[c_i] = (mu_c, C_c)

    def __call__(self, D, p=None, log_domain: bool = True):
        if p == None:
            # uniform priors
            p = col(np.full_like(self.c, 1/len(self.c), np.float32))
        if len(self.c) > 2:
            # multiclass
            Sjoint = self._Sjoint(D, p, log_domain)
            Smarg = self._Smarg(Sjoint, log_domain)
            Spost = self._Spost(Sjoint, Smarg)
            return Spost
        else:
            # binary
            llr = self._llr(D)
            return llr

    def _llr(self, D):
        S = []
        for c_i in self.c:
                mu, C = self.params[c_i]
                S.append(logpdf_gau_nd(D, mu, C))
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


def main():
    # Iris dataset loading and train-eval split
    D, l = load_iris()
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)

    # MULTICLASS PROBLEM 
    # Multivariate Gaussian Classifier
    mvg = MVG()
    #   Training: ML estimates
    mvg.fit(Dtr, ltr)
    #   Inference
    c = np.unique(l)
    p = col(np.full_like(c, 1/len(c), np.float32)) # priors
    #       NO log-domain
    Sjoint = mvg._Sjoint(Dval, p, log_domain=False)
    assert(np.allclose(Sjoint, np.load('solution/SJoint_MVG.npy')))
    Smarg = mvg._Smarg(Sjoint, log_domain=False)
    Spost = mvg._Spost(Sjoint, Smarg, log_domain=False)
    assert(np.allclose(Spost, np.load('solution/Posterior_MVG.npy')))
    #       Log-domain
    logSjoint = mvg._Sjoint(Dval, p)
    assert(np.allclose(logSjoint, np.load('solution/logSJoint_MVG.npy')))
    logSmarg = mvg._Smarg(logSjoint)
    assert(np.allclose(logSmarg, np.load('solution/logMarginal_MVG.npy')))
    logSpost = mvg._Spost(logSjoint, logSmarg)
    assert(np.allclose(logSpost, np.load('solution/logPosterior_MVG.npy')))
    assert(np.allclose(Spost, np.exp(logSpost)))
    preds = np.argmax(Spost, axis=0)
    acc = np.sum(preds==lval) / len(lval)
    err = 1 - acc
    print('Multiclass Error rates:')
    print(f'\tMVG: {err*100:.2f}%')

    # Naive Bayes Gaussian Classifier
    mvg_nb = NaiveBayesMVG()
    #   Training: ML estimates
    mvg_nb.fit(Dtr, ltr)
    #   Inference (log-domain)
    logSjoint = mvg_nb._Sjoint(Dval, p)
    assert(np.allclose(logSjoint, np.load('solution/logSJoint_NaiveBayes.npy')))
    logSmarg = mvg_nb._Smarg(logSjoint)
    assert(np.allclose(logSmarg, np.load('solution/logMarginal_NaiveBayes.npy')))
    logSpost = mvg_nb._Spost(logSjoint, logSmarg)
    assert(np.allclose(logSpost, np.load('solution/logPosterior_NaiveBayes.npy')))
    Spost = np.exp(logSpost)
    preds = np.argmax(Spost, axis=0)
    acc = np.sum(preds==lval) / len(lval)
    err = 1 - acc
    print(f'\tNaive Bayes MVG: {err*100:.2f}%')

    # Tied Covariance Gaussian Classifier
    mvg_tc = TiedCovarianceMVG()
    #   Training: ML estimates
    mvg_tc.fit(Dtr, ltr)
    #   Inference (log-domain)
    logSjoint = mvg_tc._Sjoint(Dval, p)
    assert(np.allclose(logSjoint, np.load('solution/logSJoint_TiedMVG.npy')))
    logSmarg = mvg_tc._Smarg(logSjoint)
    assert(np.allclose(logSmarg, np.load('solution/logMarginal_TiedMVG.npy')))
    logSpost = mvg_tc._Spost(logSjoint, logSmarg)
    assert(np.allclose(logSpost, np.load('solution/logPosterior_TiedMVG.npy')))
    Spost = np.exp(logSpost)
    preds = np.argmax(Spost, axis=0)
    acc = np.sum(preds==lval) / len(lval)
    err = 1 - acc
    print(f'\tTied Covariance MVG: {err*100:.2f}%')

    # BINARY PROBLEM
    D = D[:, l!=0]
    l = l[l!=0]
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)

    # Multivariate Gaussian Classifier
    mvg = MVG()
    #   Training: ML estimates
    mvg.fit(Dtr, ltr)
    #   Inference
    llr_D = mvg._llr(Dval)
    assert(np.allclose(llr_D, np.load('solution/llr_MVG.npy')))
    p = col(np.full_like(np.unique(lval), 1/2, np.float32))
    t = -np.log(p[1] / p[0])
    preds = [2 if llr_D[i] >= t else 1 for i in range(len(llr_D))]
    acc = np.sum(preds == lval) / len(lval)
    err = 1 - acc
    print('Binary Error rates:')
    print(f'\tMVG: {err*100:.2f}%')

    # Tied Covariance Gaussian Classifier
    mvg_tc = TiedCovarianceMVG()
    #   Training: ML estimates
    mvg_tc.fit(Dtr, ltr)
    #   Inference
    llr_D = mvg_tc._llr(Dval)
    p = col(np.full_like(np.unique(lval), 1/2, np.float32))
    t = -np.log(p[1] / p[0])
    preds = [2 if llr_D[i] >= t else 1 for i in range(len(llr_D))]
    acc = np.sum(preds == lval) / len(lval)
    err = 1 - acc
    print(f'\tTied Covariance MVG: {err*100:.2f}%')
    #   Verify that the Tied Covariance MVG provides the same labels as LDA
    W = lda(Dtr, ltr, 1)
    Dtr_lda = W.T @ Dtr
    if np.mean(Dtr_lda[:, ltr==1]) > np.mean(Dtr_lda[:, ltr==2]): # fix orientation if needed 
        W = -W
        Dtr_lda = W.T @ Dtr_lda
    t = (np.mean(Dtr_lda[:, ltr==1]) + np.mean(Dtr_lda[:, ltr==2])) / 2
    Dval_lda = W.T @ Dval
    pval_lda = np.array([2 if x >= t else 1 for x in Dval_lda.ravel()])
    assert(np.allclose(pval_lda, preds))
    acc = np.sum(pval_lda == lval) / len(lval)
    err = 1 - acc
    print(f'\tLDA: {err*100:.2f}%')
    

if __name__ == '__main__':
    main()