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
    

def plot_pca(data, labels):
    plt.figure()
    for label in np.unique(labels):
        mask = (labels == label)
        plt.scatter(data[0, mask], data[1, mask], label=label_names[label])
    plt.xlabel('First direction')
    plt.ylabel('Second direction')
    plt.legend()
    plt.title('PCA')
    plt.savefig(f'PCA.png')

def plot_lda(data, labels):
    plt.figure()
    for label in np.unique(labels):
        mask = (labels == label)
        plt.scatter(data[0, mask], data[1, mask], label=label_names[label])
    plt.xlabel('First direction')
    plt.ylabel('Second direction')
    plt.legend()
    plt.title('LDA')
    plt.savefig(f'LDA.png')

def plot_hist(data, labels, name: str):
    plt.figure()
    for label in np.unique(labels):
        mask = (labels == label)
        plt.hist(data[0, mask], label=label_names[label], bins=5, density=True, alpha=0.4, ec='black')
    plt.legend()
    plt.title(name)
    plt.savefig(f'{name}.png')
        
def main():
    data, labels = load_data()
    P = pca(data, 2)
    #data_pca = P.T  @ data
    #plot_pca(data_pca, labels)
    W = abs(lda(data, labels, 2))
    data_lda = W.T @ data
    plot_lda(data_lda, labels)
    """ PCA and LDA for binary classification (versicolor-virginica) """
    data = data[:, labels != 0]
    labels = labels[labels != 0]
    (Dtr, ltr), (Dval, lval) = split_2to1(data, labels)
    W = lda(Dtr, ltr, 1)
    Dtr_lda, Dval_lda = W.T @ Dtr, W.T @ Dval
    #plot_hist(Dtr_lda, ltr, 'LDA Training set')
    #plot_hist(Dval_lda, lval, 'LDA Validation set')
    # LDA error rate
    t = (Dtr_lda[:, ltr==1].mean(1) + Dtr_lda[:, ltr==2].mean(1)) / 2.0
    pred = np.array([2 if x >= t else 1 for x in Dval_lda.ravel()])
    err = (pred != lval).sum()
    print(f'LDA Error rate: {err}')
    # PCA error rate
    P = abs(pca(Dtr, 1))
    Dtr_pca, Dval_pca = P.T @ Dtr, P.T @ Dval
    t = (Dtr_pca[:, ltr==1].mean(1) + Dtr_pca[:, ltr==2].mean(1)) / 2.0
    pred = np.array([2 if x >= t else 1 for x in Dval_pca.ravel()])
    err = (pred != lval).sum()
    print(f'PCA Error rate: {err}')
    plot_hist(Dtr_pca, ltr, 'PCA Training set')
    plot_hist(Dval_pca, lval, 'PCA Validation set')
    # PCA preprocessing + LDA for classification
    #   estimation
    m = 2
    P = pca(Dtr, m)
    Dtr_pca = P.T @ Dtr
    W = lda(Dtr_pca, ltr, 1)
    Dtr_lda = W.T @ Dtr_pca
    t = (Dtr_lda[:, ltr==1].mean(1) + Dtr_lda[:, ltr==2].mean(1)) / 2.0
    #   classification
    Dval_pca = P.T @ Dval
    Dval_lda = W.T @ Dval_pca
    pred = np.array([2 if x >= t else 1 for x in Dval_lda.ravel()])
    err = (pred != lval).sum()
    print(f'PCA+LDA (m={m}) Error rate: {err}')

if __name__ == '__main__':
    main()