import numpy as np
import matplotlib.pyplot as plt
from dimensionality_reduction import pca, lda, split_2to1

label_names = {
    0: 'Genuine',
    1: 'Counterfeit'
}

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def load_data(path: str) -> np.ndarray:
    samples = []
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split(',')
            sample = col(np.array(fields[:-1], np.float32))
            samples.append(sample)
            labels.append(int(fields[-1]))
    return np.hstack(samples), np.array(labels)

def plot_feature_hist(data: np.ndarray, labels: np.ndarray, path: str):
    for i in range(data.shape[0]):
        plt.figure()
        for label in np.unique(labels):
            mask = (labels == label)
            plt.hist(data[i, mask], bins=10, density=True, 
                     alpha=0.4, label=label_names[label], ec='black')
            plt.xlabel(f'Feature {i+1}')
            plt.legend()
            plt.tight_layout()
        plt.savefig(f'{path}histogram of feature {i+1}.png')

def plot_feature_scatter(data: np.ndarray, labels: np.ndarray):
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            plt.figure()
            if i != j:
                for label in np.unique(labels):
                    mask = (labels == label)
                    plt.scatter(data[i, mask], data[j, mask], label=label_names[label])
                    plt.xlabel(f'Feature {i+1}')
                    plt.ylabel(f'Feature {j+1}')
                    plt.legend()
                    plt.tight_layout()
                plt.savefig(f'./plots/scatter plot of feature {i+1}-{j+1}.png')

def main():
    data, labels = load_data('train.txt')
    #plot_feature_hist(data, labels, './plots/')
    #plot_feature_scatter(data, labels)
    mean = np.mean(data, 1, keepdims=True)
    #data_c = data - mean
    with open('dataset_stats.txt', 'w') as f:
        print(f'Dataset mean:\n {mean}', file=f)
        cov = (data @ data.T) / data.shape[1]
        print(f'Dataset covariance:\n {cov}', file=f)
        var = np.var(data, 1, keepdims=True)
        print(f'Dataset variance:\n {var}', file=f)
        std = np.std(data, 1, keepdims=True)
        print(f'Dataset standard deviation:\n {std}', file=f)
    for label_enc in np.unique(labels):
        with open(f'{label_names[label_enc]}_stats.txt', 'w') as f:
            print(f'{label_names[label_enc]} statistics:', file=f)
            mask = (labels == label_enc)
            data_l = data[:, mask]
            mean = np.mean(data_l, 1, keepdims=True)
            print(f'mean:\n {mean}', file=f)
            cov = (data_l @ data_l.T) / data_l.shape[1]
            print(f'covariance:\n {cov}', file=f)
            var = np.var(data_l, 1, keepdims=True)
            print(f'variance:\n {var}', file=f)
            std = np.std(data_l, 1, keepdims=True)
            print(f'standard deviation:\n {std}', file=f)
    """ PCA and LDA """
    m = 6
    P = pca(data, m)
    D_pca = P.T @ data
    #plot_feature_hist(D_pca, labels, './plots/PCA/')
    W = lda(data, labels, 1)
    D_lda = W.T @ data
    plot_feature_hist(D_lda, labels, './plots/LDA/')
    # using LDA as a classifier
    (Dtr, ltr), (Dval, lval) = split_2to1(data, labels)
    W = lda(Dtr, ltr, 1)
    W = W if np.mean((W.T @ Dtr)[0, ltr==1]) > np.mean((W.T @ Dtr)[0, ltr==0]) else -W
    Dtr_lda = W.T @ Dtr
    t = (Dtr_lda[:, ltr==1].mean(1) + Dtr_lda[:, ltr==0].mean(1)) / 2.0
    #t += -0.2
    Dval_lda = W.T @ Dval
    pred = np.array([1 if x > t else 0 for x in Dval_lda.ravel()])
    err = (pred != lval).sum()
    print(f'LDA Error rate: {err / Dval_lda.shape[1]}')
    #   preprocessing using PCA
    m = 6
    errs = {}
    for m in range(1, 7):
        P = pca(Dtr, m)
        Dtr_pca = P.T @ Dtr
        W = lda(Dtr_pca, ltr, 1)
        W = W if np.mean((W.T @ Dtr_pca)[0, ltr==1]) > np.mean((W.T @ Dtr_pca)[0, ltr==0]) else -W
        Dtr_lda = W.T @ Dtr_pca
        t = (Dtr_lda[:, ltr==1].mean(1) + Dtr_lda[:, ltr==0].mean(1)) / 2.0
        #   classification
        Dval_pca = P.T @ Dval
        Dval_lda = W.T @ Dval_pca
        pred = np.array([1 if x > t else 0 for x in Dval_lda.ravel()])
        err = (pred != lval).sum()
        errs[str(m)] = err / Dval_lda.shape[1]
    print('LDA+PCA:', errs)


if __name__ == '__main__':
    main()