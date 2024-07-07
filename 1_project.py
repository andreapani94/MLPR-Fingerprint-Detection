from libs.utils import load_data, plot_feature_hist, plot_feature_scatter
from libs.utils import print_table, print_matrix, label_names
import numpy as np

def main():
    data, labels = load_data('dataset/train.txt')
    # histogram for every feature
    plot_feature_hist(data, labels, path='plots')
    # scatter-plot for every feature-pair
    plot_feature_scatter(data, labels, path='plots')
    # compare class-means for every feature
    #   clear file
    with open('results/statistics.txt', 'w') as f:
        f.truncate()
    table_rows = []
    for i in range(data.shape[0]):
        mu_0 = np.mean(data[i, labels==0])
        mu_1 = np.mean(data[i, labels==1])
        table_rows.append((f'Feature {i+1}', mu_0, mu_1))
    print_table(table_rows, headers=['', label_names[0], label_names[1]], 
                title='Class means comparison', filepath='results/statistics.txt')
    table_rows.clear()
    # compare class-variances for every feature
    for i in range(data.shape[0]):
        var_0 = np.var(data[i, labels==0])
        var_1 = np.var(data[i, labels==1])
        table_rows.append((f'Feature {i+1}', var_0, var_1))
    print_table(table_rows, headers=['', label_names[0], label_names[1]], 
                title='Class variances comparison', filepath='results/statistics.txt')
    # compare covariance matrices for each class
    C_0 = np.cov(data[:, labels==0])
    C_1 = np.cov(data[:, labels==1])
    print_matrix(C_0, title=f'{label_names[0]} covariance', filepath='results/statistics.txt')
    print_matrix(C_1, title=f'{label_names[1]} covariance', filepath='results/statistics.txt')

if __name__ == '__main__':
    main()