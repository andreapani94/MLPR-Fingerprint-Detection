from libs.utils import load_data, label_names, row
from libs.density_estimation import logpdf_gau_nd, gau_nd_ml_estimates
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle


def main():
    data, labels = load_data('dataset/train.txt')
    # plot the estimated density on top of the histogram
    # for every feature and label
    for i in range(data.shape[0]):
        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        plt.figure()
        for label in np.unique(labels):
            color = next(colors)
            plt.hist(data[i, labels==label], bins=30, density=True, alpha=0.4, 
                        label=label_names[label], ec='black', color=color)
            # compute ml estimates for this feature and label
            mu, C = gau_nd_ml_estimates(row(data[i, labels==label]))
            xplot = np.linspace(-4, 4, 1000)
            yplot = np.exp(logpdf_gau_nd(row(xplot), mu, C))
            plt.plot(xplot, yplot, linewidth=2, color=color)
        plt.legend()
        plt.title(f'Feature {i+1}')
        plt.savefig(f'plots/fit_gaupdf_feature{i+1}.png')
        plt.close()



if __name__ == '__main__':
    main()