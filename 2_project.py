from libs.utils import load_data, plot_feature_hist, split_2to1, print_table, label_names
from libs.dimensionality_reduction import PCA, LDA
import numpy as np

def find_best_threshold(scores, labels, init_value):
    thresholds = np.linspace(init_value - 1, init_value + 1, 10000)
    threshold_error_map = {}

    for threshold in thresholds:
        predictions = np.where(scores > threshold, 1, 0)
        threshold_error_map[threshold] = np.sum(predictions != labels) / len(labels)
    #       find the best threshold
    best_threshold = min(threshold_error_map, key=threshold_error_map.get)

    return best_threshold

def main():
    data, labels = load_data('dataset/train.txt')
    # plot the projected PCA features
    pca = PCA(n_components=6)
    pca.fit(data)
    data_pca = pca(data)
    plot_feature_hist(data_pca, labels, 'plots', 'PCA')
    # # compare class-variances for every feature
    # table_rows = []
    # for i in range(data_pca.shape[0]):
    #     var_0 = np.var(data_pca[i, labels==0])
    #     var_1 = np.var(data_pca[i, labels==1])
    #     table_rows.append((f'Feature {i+1}', var_0, var_1))
    # print_table(table_rows, headers=['', label_names[0], label_names[1]], 
    #             title='Class variances comparison')
    # plot the projected LDA feature
    lda = LDA(n_components=1)
    lda.fit(data, labels, fix_orientation=True)
    data_lda = lda(data)
    plot_feature_hist(data_lda, labels, 'plots', 'LDA')
    # LDA for classification
    #   clear file
    with open('results/dimensionality_reduction.txt', 'w') as f:
        f.truncate()
    table_rows = []
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    lda = LDA(n_components=1)
    lda.fit(train_data, train_labels, fix_orientation=True)
    train_data_lda = lda(train_data)
    lda_threshold = (np.mean(train_data_lda[:, train_labels==1]) + 
                    np.mean(train_data_lda[:, train_labels==0])) / 2
    scores = lda(val_data)
    predictions = np.where(scores >= lda_threshold, 1, 0)
    error = np.sum(predictions != val_labels) / len(val_labels)
    #   trying different thresholds
    best_threshold = find_best_threshold(scores, val_labels, lda_threshold)
    predictions = np.where(scores >= best_threshold, 1, 0)
    best_error = np.sum(predictions != val_labels) / len(val_labels)
    table_rows.append(('No PCA', lda_threshold, f'{error*100:.2f}%', best_threshold, f'{best_error*100:.2f}%'))
    #   PCA pre-processing + LDA for classification
    for n_components in np.arange(2, 6):
        # estimation
        pca = PCA(n_components)
        pca.fit(train_data)
        train_data_pca = pca(train_data)
        lda = LDA(n_components=1)
        lda.fit(train_data_pca, train_labels, fix_orientation=True)
        train_data_lda = lda(train_data_pca)
        lda_threshold = (np.mean(train_data_lda[:, train_labels==1]) + 
                    np.mean(train_data_lda[:, train_labels==0])) / 2
        # classification
        val_data_pca = pca(val_data)
        scores = lda(val_data_pca)
        best_threshold = find_best_threshold(scores, val_labels, lda_threshold)
        predictions = np.where(scores >= lda_threshold, 1, 0)
        error1 = np.sum(predictions != val_labels) / len(val_labels)
        predictions = np.where(scores >= best_threshold, 1, 0)
        error2 = np.sum(predictions != val_labels) / len(val_labels)
        table_rows.append((f'{n_components}', lda_threshold, f'{error1*100:.2f}%', best_threshold, f'{error2*100:.2f}%'))
    print_table(table_rows, ['PCA components', 'LDA threshold', 'Error rate', 'Best threshold', 'Error rate'], 'PCA+LDA classification')
    

if __name__ == '__main__':
    main()