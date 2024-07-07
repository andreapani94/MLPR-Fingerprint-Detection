from libs.utils import load_data, split_2to1, print_table, to_snake_case
from libs.utils import print_matrix, col, row, label_names
from libs.gaussian_models import MVG, NaiveBayesMVG, TiedCovarianceMVG
from libs.dimensionality_reduction import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tabulate import tabulate

filepath = 'results/gaussian_models.txt'

def corr(C: np.ndarray):
    return C / (col(C.diagonal()**0.5) * row(C.diagonal()**0.5))

def main():
    data, labels = load_data('dataset/train.txt')
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    table_rows = []
    #   clear results file
    with open(filepath, 'w') as f:
        f.truncate()
    # apply the MVG model to the dataset
    mvg, mvg_tc, mvg_nb = MVG(), TiedCovarianceMVG(), NaiveBayesMVG()
    for title, model in [('MVG', mvg), ('Tied Covariance MVG', mvg_tc), ('Naive Bayes MVG', mvg_nb)]:
        model.fit(train_data, train_labels)
        llr_scores = model(val_data)
        predictions = np.where(llr_scores >= 0, 1, 0)
        error = 1 - (np.sum(predictions == val_labels) / len(val_labels))
        table_rows.append((title, f'{error*100:.2f}%'))
        np.save(f'results/gaussian_models/{to_snake_case(title)}_llr', llr_scores)
    print_table(table_rows, headers=['', 'Error rate'], filepath=filepath)
    # correlation analysis
    for label in [0, 1]:
        C = mvg.params[label][1]
        print(f'{label_names[label]}:')
        print_matrix(C, 'Covariance', filepath=filepath)
        C_corr = corr(C)
        print_matrix(C_corr, 'Correlation', filepath=filepath)
        sn.heatmap(C_corr, linewidths=2, cmap='Reds')
        plt.savefig(f'plots/heatmap_corr_{to_snake_case(label_names[label])}.png')
        plt.close()
    # MVG models on a fraction of features (1 to 4), (1-2), (3-4)
    for feature_idxs in [np.arange(4), [0, 1], [2, 3]]:
        table_rows.clear()
        (train_data, train_labels), (val_data, val_labels) = split_2to1(data[feature_idxs], labels)
        for title, model in [('MVG', mvg), ('Tied Covariance MVG', mvg_tc), ('Naive Bayes MVG', mvg_nb)]:
            model.fit(train_data, train_labels)
            llr_scores = model(val_data)
            predictions = np.where(llr_scores >= 0, 1, 0)
            error = 1 - (np.sum(predictions == val_labels) / len(val_labels))
            table_rows.append((title, f'{error*100:.2f}%'))
            #np.save(f'results/gaussian_models/{to_snake_case(title)}_llr', llr_scores)
        print_table(table_rows, headers=['', 'Error rate'], title='Error rates using features ' 
                        + ''.join([f'{n+1} ' for n in feature_idxs]), filepath=filepath)
    # analyze PCA as preprocessing + MVG models
    table_rows.clear()
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    for n_components in np.arange(2, 6):
        pca = PCA(n_components)
        pca.fit(train_data)
        train_data_pca = pca(train_data)
        val_data_pca = pca(val_data)
        for title, model in [('MVG', mvg), ('Tied Covariance MVG', mvg_tc), ('Naive Bayes MVG', mvg_nb)]:
            model.fit(train_data_pca, train_labels)
            llr_scores = model(val_data_pca)
            predictions = np.where(llr_scores >= 0, 1, 0)
            error = 1 - (np.sum(predictions == val_labels) / len(val_labels))
            table_rows.append((str(n_components), title, f'{error*100:.2f}%'))
            np.save(f'results/gaussian_models/{to_snake_case(title)}_pca(m={n_components})_llr', llr_scores)
    print_table(table_rows, headers=['PCA components', 'Model', 'Error rate'], 
                    title='PCA + MVG models', filepath=filepath)

        


if __name__ == '__main__':
    main()