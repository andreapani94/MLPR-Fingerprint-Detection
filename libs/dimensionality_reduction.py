import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import scipy.linalg

label_encodings = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}

label_names = {v:k for k,v in label_encodings.items()}

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def load_data():
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']

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
        plt.hist(data[feat_idx, mask], label=label_names[label], bins=5, density=True, alpha=0.4, ec='black')
    plt.legend()
    plt.title(title.upper())
    plt.savefig(f'{title}_feature{feat_idx+1}.png')

class PCA:
    def __init__(self, n_components: int=None):
        self.n_components = n_components

    def fit(self, X: np.ndarray):
        if not self.n_components:
            self.n_components = X.shape[0]
        self.projection_matrix = pca(X, self.n_components)

    def __call__(self, X: np.ndarray):
        P = self.projection_matrix
        return P.T @ X
    
class LDA:
    def __init__(self, n_components: int=None):
        self.n_components = n_components

    def fit(self, X: np.ndarray, y, fix_orientation=False):
        self.projection_matrix = lda(X, y, self.n_components)
        if fix_orientation and len(np.unique(y)) == 2:
            self._fix_orientation(X, y)

    def __call__(self, X: np.ndarray):
        P = self.projection_matrix
        return P.T @ X
    
    def _fix_orientation(self, X, y):
        Xp = self(X)
        if np.mean(Xp[:, y==0]) > np.mean(Xp[:, y==1]):
            self.projection_matrix = -self.projection_matrix
    

        
def main():
    D, l = load_data()

    # DIMENSIONALITY REDUCTION
    #   PRINCIPAL COMPONENT ANALYSIS (PCA)
    P = pca(D, m=4)
    assert(np.allclose(P, np.load('solution/IRIS_PCA_matrix_m4.npy')))
    Dpca = P[:, :2].T @ D
    plot_data(Dpca, l, 'pca')
    plot_featurehist(Dpca, l, 0, 'pca')
    #   LINEAR DISCRIMINANT ANALYSIS (LDA)
    W = lda(D, l, 2)
    assert(np.allclose(W, np.load('solution/IRIS_LDA_matrix_m2.npy')))
    Dlda = W.T @ D
    plot_data(Dlda, l, 'lda')
    plot_featurehist(Dlda, l, 0, 'lda')

    # PCA and LDA for CLASSIFICATION
    #   LDA without pre-processing
    #       binary dataset including only labels 1 and 2
    D = D[:, l!=0]
    l = l[l!=0]
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    W = lda(Dtr, ltr, 1)
    Dtr_lda, Dval_lda = W.T @ Dtr, W.T @ Dval
    plot_featurehist(Dtr_lda, ltr, 0, 'lda_train')
    plot_featurehist(Dval_lda, lval, 0, 'lda_eval')
    #       compute the LDA classification threshold
    if np.mean(Dtr_lda[:, ltr==1]) > np.mean(Dtr_lda[:, ltr==2]): # fix orientation if needed 
        W = -W
        Dtr_lda = W.T @ Dtr_lda
    t = (np.mean(Dtr_lda[:, ltr==1]) + np.mean(Dtr_lda[:, ltr==2])) / 2
    preds = np.array([2 if x >= t else 1 for x in Dval_lda.ravel()])
    err = (preds != lval).sum()
    print(f'LDA error rate: {err}/{len(lval)}')
    #       confront with PCA
    P = pca(Dtr, m=1)
    Dtr_pca, Dval_pca = P.T @ Dtr, P.T @ Dval
    if np.mean(Dtr_pca[:, ltr==1]) > np.mean(Dtr_pca[:, ltr==2]): # fix orientation if needed 
        P = -P
        Dtr_pca = P.T @ Dtr
    t = (np.mean(Dtr_pca[:, ltr==1]) + np.mean(Dtr_pca[:, ltr==2])) / 2
    preds = np.array([2 if x >= t else 1 for x in Dval_pca.ravel()])
    err = (preds != lval).sum()
    print(f'PCA error rate: {err}/{len(lval)}')
    #   LDA + PCA pre-processing
    m = 2
    #       PCA estimation
    P = pca(Dtr, m)
    Dtr_pca = P.T @ Dtr
    #       LDA estimation
    W = lda(Dtr_pca, ltr, 1)
    Dtr_lda = W.T @ Dtr_pca
    if np.mean(Dtr_lda[:, ltr==1]) > np.mean(Dtr_lda[:, ltr==2]): # fix orientation if needed 
        W = -W
        Dtr_lda = W.T @ Dtr_pca
    t = (np.mean(Dtr_lda[:, ltr==1]) + np.mean(Dtr_lda[:, ltr==2])) / 2
    #       Classification pipeline
    Dval_pca = P.T @ Dval
    Dval_lda = W.T @ Dval_pca
    preds = np.array([2 if x >= t else 1 for x in Dval_lda.ravel()])
    err = (preds != lval).sum()
    print(f'LDA+PCA with m={m} error rate: {err}/{len(lval)}')
         
if __name__ == '__main__':
    main()