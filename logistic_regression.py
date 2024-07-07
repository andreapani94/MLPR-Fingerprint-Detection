from scipy.optimize import fmin_l_bfgs_b
import sklearn.datasets
import numpy as np
from numpy.linalg import norm
from scipy.special import logsumexp
from tabulate import tabulate
from model_evaluation import DCF, DCF_min, confusion_matrix, bayes_pred_llr

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def load_iris():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

def load_iris_bin():
    D, l = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, l!=0] # We remove setosa from D
    l = l[l!=0] # We remove setosa from L
    l[l==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, l

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

def f(x: np.ndarray):
    return (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2

def f_withgrad(x: np.ndarray):
    grad = np.array([
        2 * (x[0] + 3) + np.cos(x[0]),
        2 * (x[1] + 1)
    ])
    y = (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2
    return y, grad

def print_logreg_res(data):
    perc_format = lambda x: f'{x*100:.2f}%'
    headers = ["λ", "J(w*, b*)", "Error rate", "minDCF", "actDCF"]
    data_format = [
        [row[0], row[1], perc_format(row[2]), row[3], row[4]] for row in data
    ]
    print(tabulate(data_format, headers, tablefmt="grid", floatfmt='.4f'))
    print()

class LogisticRegression:
    def __init__(self, l: float=0, pi=None):
        self.l = l
        if pi != None:
            self.pi = pi
            self.prior_weighted = True
        else:
            self.prior_weighted = False

    def train(self, X, y, return_opt=False):
        # initialize weights and bias
        x0 = np.zeros(X.shape[0] + 1)
        # optimize using the L-BFGS algorithm
        x_opt, obj_opt, info = fmin_l_bfgs_b(self.logreg_obj, x0, args=[X, y, self.l])
        # save parameters 
        w, b = col(x_opt[:-1]), x_opt[-1]
        self.params = (w, b)
        if return_opt:
            return x_opt, obj_opt, info

    def __call__(self, X):
        w, b = self.params
        return ((w.T @ X) + b).ravel()

    def logreg_obj(self, params, X, y, l):
        w, b = params[:-1], params[-1]
        z = (2 * y) - 1
        S = ((col(w).T @ X) + b).ravel()
        if not self.prior_weighted:
            f = (l / 2) * norm(w)**2 + np.mean(np.logaddexp(0, -z * S))
            G = -z / (1 + np.exp(z * S))
            f_grad = np.hstack([((l * w) + np.mean(row(G) * X, 1)).ravel(), np.mean(G)])
        else:
            N1, N0 = X[:, y==1].shape[1], X[:, y==0].shape[1]
            xi = np.where(z==1, self.pi / N1, (1 - self.pi) / N0)
            f = (l / 2) * norm(w)**2 + np.sum(xi * np.logaddexp(0, -z * S))
            G = -z / (1 + np.exp(z * S))
            f_grad = np.hstack([((l * w) + np.sum(xi * row(G) * X, 1)).ravel(), np.sum(xi * G)])
        return f, f_grad
    
    def logreg_obj_mul(self, params):
        W, b = params[:-1], params[-1]
        Dtr, ltr, l = self.Dtrain, self.ltrain, self.reg_param
        D, N, K = Dtr.shape[0], Dtr.shape[1], len(np.unique(ltr))
        W = np.reshape(W, (D, K))
        S = (W.T @ Dtr) + b
        # one hot encoding
        T = np.zeros((K, N), dtype=np.int32)
        T[ltr, np.arange(N)] += 1
        # numerically stable softmax
        Y = S - logsumexp(S, axis=0)
        y = (l / 2) * norm(W)**2 - ((1 / N) * np.sum(T * Y))
        return y

def main():
    # NUMERICAL OPTIMIZATION
    #   with gradient approximated
    x0 = np.array([0, 0])
    x_opt, f_opt, info = fmin_l_bfgs_b(f, x0, approx_grad=True)
    assert(np.allclose(x_opt, np.array([-2.57747138, -0.99999927])))
    print('L-BFGS-B with gradient estimation:')
    print(f"  x_min={x_opt}, y_min={f_opt}, n_steps={info['funcalls']}")
    #   with provided gradient
    x_opt, f_opt, info = fmin_l_bfgs_b(f_withgrad, x0)
    assert(np.allclose(x_opt, np.array([-2.57747138, -0.99999927])))
    print('L-BFGS-B without gradient estimation:')
    print(f"  x_min={x_opt}, y_min={f_opt}, n_steps={info['funcalls']}")

    # BINARY LOGISTIC REGRESSION
    D, l = load_iris_bin()
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    lam_values = [0.001, 0.1, 1]
    pi = 0.5
    pi_emp = Dtr[:, ltr==1].shape[1] / Dtr.shape[1]
    logreg_res = []
    for lam in lam_values:
        logreg = LogisticRegression(lam)
        # optimize model parameters (training)
        _, obj_opt, _ = logreg.train(Dtr, ltr, return_opt=True)
        # Evaluation 
        S = logreg(Dval)
        lpred = np.where(S > 0, 1, 0)
        err = 1 - (np.sum(lpred == lval) / len(lval))
        Sllr = S - np.log(pi_emp / (1 - pi_emp))
        lpred = bayes_pred_llr(Sllr, pi, 1, 1)
        M = confusion_matrix(lpred, lval)
        actDCF = DCF(M, pi)
        minDCF = DCF_min(Sllr, lval, pi)
        logreg_res.append((lam, obj_opt, err, minDCF, actDCF))
    print(f'\nLogistic regression with π = {pi}')
    print_logreg_res(logreg_res)

    # PRIOR-WEIGHTED LOGISTIC REGRESSION AND CALIBRATION
    logreg_res.clear()
    pi = 0.8
    for lam in lam_values:
        logreg = LogisticRegression(lam, pi=pi)
         # optimize model parameters (training)
        _, obj_opt, _ = logreg.train(Dtr, ltr, return_opt=True)
        # Evaluation
        S = logreg(Dval)
        lpred = np.where(S > 0, 1, 0)
        err = 1 - (np.sum(lpred == lval) / len(lval))
        Sllr = S - np.log(pi / (1 - pi))
        lpred = bayes_pred_llr(Sllr, pi, 1, 1)
        M = confusion_matrix(lpred, lval)
        actDCF = DCF(M, pi)
        minDCF = DCF_min(Sllr, lval, pi)
        logreg_res.append((lam, obj_opt, err, minDCF, actDCF))
    print(f'\nPrior-weighted Logistic regression with π = {pi}')
    print_logreg_res(logreg_res)

    # # MULTICLASS LOGISTIC REGRESSION
    # D, l = load_iris()
    # (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    # logreg_res = []
    # for lam in lam_values:
    #     logreg = LogisticRegression(lam)
    #     D, K = Dtr.shape[0], len(np.unique(ltr))
    #     x0 = np.zeros((D * K) + 1)
    #     par_opt, obj_opt, _ = fmin_l_bfgs_b(logreg.logreg_obj_mul, x0, approx_grad=True)
    #     W, b = np.reshape(par_opt[:-1], (D, K)), par_opt[-1]
    #     S = (W.T @ Dval) + b
    #     lpred = np.argmax(S, 0)
    #     err = 1 - (np.sum(lpred == lval) / len(lval))
    #     print(lam, obj_opt, err)

if __name__ == '__main__':
    main()

