import numpy as np
import matplotlib.pyplot as plt
from libs.utils import load_data, col, row

label_names = {
    0: 'Genuine',
    1: 'Counterfeit'
}

def logpdf_GAU_ND_x(x: np.ndarray, mu, C):
    """
        x: data sample (Mx1)
        mu: mean (Mx1)
        C: covariance matrix
    """
    logN = 0
    M = x.shape[0]
    logN -= (M/2) * np.log(2*np.pi)
    _ , logdetC = np.linalg.slogdet(C)
    logN -= (0.5) * logdetC
    logN -= (0.5) * np.dot(np.dot((x - mu).T, np.linalg.inv(C)), (x - mu)).ravel()
    return logN

def logpdf_gau_nd(X, mu, C):
    M = X.shape[0]
    _, logdetC = np.linalg.slogdet(C)
    C_inv = np.linalg.inv(C)
    const = -(M/2) * np.log(2*np.pi) - 1/2 * logdetC
    Y = const -1/2 * np.sum((X - mu) * (C_inv @ (X - mu)), 0)
    return Y.ravel()

def main():
    D, l = load_data('dataset/train.txt')
    for i in range(D.shape[0]):
        plt.figure()
        for lab in label_names:
            Dlab = D[:, l==lab]
            mu_ML = col(np.mean(Dlab[i]))
            C_ML = ((Dlab[i] - mu_ML) @ (Dlab[i] - mu_ML).T) / Dlab.shape[1]
            plt.hist(Dlab[i], bins=30, density=True, alpha=0.4, label=label_names[lab], ec='black')
            xplot = np.linspace(-4, 4, 1000)
            yplot = np.exp(logpdf_gau_nd(row(xplot), mu_ML, C_ML))
            plt.plot(xplot, yplot, linewidth=2)
        plt.legend()
        plt.title(f'Feature {i+1}')
        plt.savefig(f'plots/fit_gaupdf_feature{i+1}.png')

if __name__ == '__main__':
    main()
    
    
    """
    mu = np.array([[0.0]])
    C = np.array([[2.0]])
    X1 = np.random.normal(mu.ravel(), C.ravel(), 1000)
    plt.hist(X1, bins=30, density=True)
    X2 = np.linspace(-6, 6, 1000)
    pdf = np.exp(logpdf_gau_nd(row(X2), mu, C))
    #pdf = 1/(C * np.sqrt(2 * np.pi)) * np.exp( - (X2 - mu)**2 / (2 * C**2) )
    plt.plot(X2, pdf.ravel())
    x = np.array([[4.0]])
    print(np.exp(logpdf_gau_nd(row(x), mu, C)))
    print(1/(C * np.sqrt(2 * np.pi)) * np.exp( - (x - mu)**2 / (2 * C**2) ))
    plt.show()"""