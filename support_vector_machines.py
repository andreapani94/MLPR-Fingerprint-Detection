import numpy as np
from numpy.linalg import norm
import sklearn.datasets
from scipy.optimize import fmin_l_bfgs_b
from tabulate import tabulate
from model_evaluation import DCF, DCF_min, confusion_matrix

K_vals = [1, 10]
C_vals = [0.1, 1, 10]

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

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

def obj_primal(D: np.ndarray, z: np.ndarray, a: np.ndarray, C: int):
    w_ext = col(np.sum(a * z * D, axis=1))
    S = 1 - (z * (w_ext.T @ D))
    y = (0.5 * norm(w_ext)**2) + (C * np.sum(np.maximum(0, S)))
    return w_ext, y

def obj_dual(a: np.ndarray, D: np.ndarray, z: np.ndarray, kern=None, K=1):
    N: int = D.shape[1]
    a = col(a)
    # # slow version
    # H_slow = np.zeros((N, N))
    # for i in range(N):
    #     for j in range(N):
    #         if kern is None:
    #             H_slow[i, j] += z[i] * z[j] * (D[:, i].T @ D[:, j])
    #         else:
    #             H_slow[i, j] += z[i] * z[j] * (kern(D[:, i], D[:, j]) + K)
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
            w, b = self.params
            S = w.T @ X + b
            return S.ravel()
        else:
            a, z = self.params
            Skernel = self.kernel(self.Xtrain, X) + self.K
            return np.sum(a * z * Skernel, 0).ravel()



    def train(self, X, y, return_opt=False):
        N = X.shape[1]
        x0 = np.zeros(N)
        bounds = [(0, self.C) for _ in range(N)]
        # extend the features
        X_ext = np.vstack([X, np.full((1, N), self.K)])
        # encode the labels as 1, -1
        z = np.where(y > 0, 1, -1)
        a_opt, dual_opt, _ = fmin_l_bfgs_b(obj_dual, args=(X_ext, z, self.kernel, self.K), x0=x0, bounds=bounds, factr=1.0)
        if self.kernel is None:
            # recover the primal solution to obtain w and b
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


def main():
    D, l = load_iris_bin()
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    # LINEAR SVM
    svm_res = []
    pi = 0.5
    for K in K_vals:
        for C in C_vals:
            svm = SVM(C, K)
            dual_opt, primal_opt = svm.train(Dtr, ltr, return_opt=True)
            duality_gap = primal_opt - dual_opt
            S = svm(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            err = 1 - (np.sum(lpred == lval) / len(lval))
            M = confusion_matrix(lpred, lval)
            dcf, dcfmin = DCF(M, pi), DCF_min(S, lval, pi)
            svm_res.append((K, C, primal_opt, dual_opt, duality_gap, err, dcf, dcfmin))
    print_svm_results(svm_res)
    # KERNEL SVM
    kernsvm_res = []
    C = 1
    for d, c in [(2, 0), (2, 1)]:
        for K in [0, 1]:
            kernsvm = SVM(C, K, poly_kernel(d, c))
            dual_opt = kernsvm.train(Dtr, ltr, return_opt=True)
            S = kernsvm(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            err = 1 - (np.sum(lpred == lval) / len(lval))
            M = confusion_matrix(lpred, lval)
            dcf, dcfmin = DCF(M, pi), DCF_min(S, lval, pi)
            kernsvm_res.append((K, C, f'Poly(d={d},c={c})', dual_opt, err, dcf, dcfmin))
    for g in [1, 10]:
        for K in [0, 1]:
            kernsvm = SVM(C, K, rbf_kernel(g))
            dual_opt = kernsvm.train(Dtr, ltr, return_opt=True)
            S = kernsvm(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            err = 1 - (np.sum(lpred == lval) / len(lval))
            M = confusion_matrix(lpred, lval)
            dcf, dcfmin = DCF(M, pi), DCF_min(S, lval, pi)
            kernsvm_res.append((K, C, f'RBF(γ={g})', dual_opt, err, dcf, dcfmin))
    print_kernsvm_results(kernsvm_res)

        
if __name__ == '__main__':
    main()