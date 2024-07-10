from libs.utils import load_data, split_2to1, to_snake_case, print_table
from libs.gaussian_mixture_models import GMM, save_gmm
from libs.model_evaluation import bayes_pred_llr, confusion_matrix, DCF, DCF_min
import numpy as np
import matplotlib.pyplot as plt

def plot_gmm_dcfs(G_values: list[int], dcfs: list[list, list], title: str, savepath: str):
    plt.figure()
    plt.plot(G_values, dcfs[0], label='actDCF', marker='o')
    plt.plot(G_values, dcfs[1], label='minDCF', marker='o', linestyle='--')
    plt.xlabel('G (Gaussian components)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.xticks(ticks=G_values)
    plt.legend()
    plt.title(title)
    plt.savefig(savepath)
    plt.close()

pi_true = 0.1
filepath = 'results/gaussian_mixture_models.txt'
 
def main():
    data, labels = load_data('dataset/train.txt')
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    #   clear results file
    with open(filepath, 'w') as f:
        f.truncate()
    G_values = 2**np.arange(0, 6)
    for title, covtype in [
        ('Full Covariance', 'full'), 
        ('Diagonal Covariance', 'diag'), 
        ('Tied Covariance', 'tied')
    ]:
        dcfs = [[], []]
        table_rows = []
        models = []
        for G in G_values:
            scores = []
            for label in [0, 1]:
                model = GMM(n_components=G, covtype=covtype)
                model.train(train_data[:, train_labels==label], conv_threshold=1e-6, covbound=0.01)
                scores.append(model(val_data))
                models.append(model)
            scores = np.vstack(scores)
            llr_scores = scores[1] - scores[0]
            predictions = bayes_pred_llr(llr_scores, pi_true)
            confmatrix = confusion_matrix(predictions, val_labels)
            actDCF = DCF(confmatrix, pi_true)
            dcfs[0].append(actDCF)
            minDCF = DCF_min(llr_scores, val_labels, pi_true)
            dcfs[1].append(minDCF)
            table_rows.append((G, actDCF, minDCF, f'{((actDCF - minDCF) / minDCF) * 100:.2f}%'))
            np.save(f'results/gmm/GMM(G={G})_{covtype}_llrs', llr_scores) 
            for gmm in models:
                save_gmm(gmm.params, f'models/gmm/GMM(G={G})_{covtype}_params')
        plot_gmm_dcfs(G_values, dcfs, f'GMM {title}', f'plots/GMM_{covtype}_G_DCFs_plot')
        print_table(table_rows, ['G', 'actDCF', 'minDCF', 'Calibration loss'], filepath=filepath, title=f'GMM {title}')
    

if __name__ == '__main__':
    main()