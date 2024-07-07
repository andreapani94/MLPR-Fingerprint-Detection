import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import scipy.linalg
from libs.utils import load_data, split_2to1

label_names = {
    0: 'Genuine',
    1: 'Counterfeit'
}

def pca(D: np.ndarray, m: int) -> np.ndarray:
    """
        Computes the PCA projection matrix 
    """
    # compute the mean
    mu = np.mean(D, axis=1, keepdims=True)
    # center the data
    Dc = D - mu
    # compute the covariance matrix
    C = (Dc @ Dc.T) / Dc.shape[1]
    # compute eigenvectors and eingenvalues
    s, U = np.linalg.eigh(C)
    # get the projection matrix
    P = U[:, ::-1][:, :m]
    return P

def lda(D: np.ndarray, l: np.ndarray, m: int) -> np.ndarray:
    """
        Computes the LDA projection matrix 
    """
    # between-class and within-class covariance matrix
    Sb = 0
    Sw = 0
    mu = np.mean(D, axis=1, keepdims=True)
    for i in np.unique(l):
        mu_c = np.mean(D[:, l==i], axis=1, keepdims=True)
        Dc = D[:, l==i]
        nc = Dc.shape[1]
        Sb += nc * ((mu_c - mu) @ (mu_c - mu).T)
        Sw += (Dc - mu_c) @ (Dc - mu_c).T
    Sb /= D.shape[1]
    Sw /= D.shape[1]
    # finding the projection matrix
    s, U = scipy.linalg.eigh(Sb, Sw)
    W = U[:, ::-1][:, :m]
    return W
    

def plot_data(data, labels, title: str):
    plt.figure()
    for label in np.unique(labels):
        mask = (labels == label)
        plt.scatter(data[0, mask], data[1, mask], label=label_names[label])
    plt.legend()
    plt.title(title.upper())
    plt.savefig(f'{title}.png')


def plot_featurehist(data, labels, feat_idx: int, title: str):
    plt.figure()
    for label in np.unique(labels):
        mask = (labels == label)
        plt.hist(data[feat_idx, mask], label=label_names[label], bins=10, density=True, alpha=0.4, ec='black')
    plt.legend()
    plt.title(title.upper())
    plt.savefig(f'plots/{title}_feature{feat_idx+1}_hist.png')

        
def main():
    D, l = load_data('dataset/train.txt')
    # apply PCA and plot the feature histograms
    P = pca(D, m=6)
    Dpca = P.T @ D
    for i in range(Dpca.shape[0]):
        plot_featurehist(Dpca, l, i, 'PCA')
    # apply LDA and plot the feature histograms
    W = lda(D, l, 1)
    Dlda = W.T @ D
    plot_featurehist(Dlda, l, 0, 'LDA')
    # apply LDA as a classifier
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    W = lda(Dtr, ltr, 1)
    #   Estimation
    Dtr_lda = W.T @ Dtr
    # fix orientation if needed
    if np.mean(Dtr_lda[:, ltr==0]) > np.mean(Dtr_lda[:, ltr==1]):
        W = -W
        Dtr_lda = W.T @ Dtr_pca
    t = (np.mean(Dtr_lda[:, ltr==1]) + np.mean(Dtr_lda[:, ltr==0])) / 2
    #   Classification
    Dval_lda = W.T @ Dval
    pval = np.array([1 if x >= t else 0 for x in Dval_lda.ravel()])
    err = (pval != lval).sum()
    print('Error rate:')
    print(f'LDA with t={t:.3f}: {(err/len(lval))*100:.2f}%')
    #       try changing the value of the threshold
    err_best, t_best = err, t
    vals = np.linspace(-0.01, 0.01, 10)
    tvals = t - vals
    for t in tvals:
        #   Classification
        Dval_lda = W.T @ Dval
        pval = np.array([1 if x >= t else 0 for x in Dval_lda.ravel()])
        err = (pval != lval).sum()
        if err < err_best:
            err_best = err 
            t_best = t 
    print(f'LDA with t={t:.3f}: {(err_best/len(lval))*100:.2f}%')
    # LDA + PCA pre-processing as a function of m
    mvals = np.arange(1, 6)
    for m in mvals:
        #   PCA estimation
        P = pca(Dtr, m)
        Dtr_pca = P.T @ Dtr
        #   LDA estimation
        W = lda(Dtr_pca, ltr, 1)
        Dtr_lda = W.T @ Dtr_pca
        # fix orientation if needed
        if np.mean(Dtr_lda[:, ltr==0]) > np.mean(Dtr_lda[:, ltr==1]):
            W = -W
            Dtr_lda = W.T @ Dtr_pca
        t = (np.mean(Dtr_lda[:, ltr==1]) + np.mean(Dtr_lda[:, ltr==0])) / 2
        #   Classification
        Dval_pca = P.T @ Dval
        Dval_lda = W.T @ Dval_pca
        pval = np.array([1 if x >= t else 0 for x in Dval_lda.ravel()])
        err = (pval != lval).sum()
        print(f'LDA with t={t:.3f} + PCA with m={m}: {(err/len(lval))*100:.2f}%')
        
if __name__ == '__main__':
    main()