from libs.utils import load_data, split_2to1, col, row, to_snake_case
from libs.support_vector_machines import SVM, poly_kernel, rbf_kernel
from libs.model_evaluation import DCF, DCF_min, confusion_matrix, bayes_pred_llr
from itertools import cycle
import numpy as np
import matplotlib.pyplot as plt

def plot_svm_dcfs(C_values: list[float], dcfs: list[list, list], title: str, savepath: str):
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(C_values, dcfs[0], label='actDCF', marker='o')
    plt.plot(C_values, dcfs[1], linestyle='--', label='minDCF', marker='o')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.title(title)
    plt.savefig(savepath)
    plt.close()

def plot_kernsvm_dcfs(C_values: list[float], param_dcfs: dict[str, list[list, list]], title: str, savepath:str):
    plt.figure()
    plt.xscale('log', base=10)
    
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
    for param in param_dcfs:
        color = next(colors)
        plt.plot(C_values, param_dcfs[param][0], color=color, label=f'actDCF ({param})', marker='o')
        plt.plot(C_values, param_dcfs[param][1], color=color, linestyle='--', label=f'minDCF ({param})', marker='o')
    
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.title(title)
    plt.savefig(savepath)
    plt.close()

pi_true = 0.1

def main():
    data, labels = load_data('dataset/train.txt')
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    #train_data, train_labels = train_data[:, ::50], train_labels[::50] # just for experiments
    C_values = np.logspace(-5, 0, 11)
    K = 1
    # analyze the DCF of the SVM as regularization changes
    #   once with the standard dataset once with the centered one
    mu = col(np.mean(train_data, 1))
    for train_data, val_data, title in [
        (train_data, val_data, 'No preprocessing'), 
        (train_data - mu, val_data - mu, 'Zero centering') 
    ]:
        dcfs = [[], []]
        for C in C_values:
            model = SVM(C, K)
            model.train(train_data, train_labels)
            scores = model(val_data)
            predictions = bayes_pred_llr(scores, pi_true)
            confmatrix = confusion_matrix(predictions, val_labels)
            actDCF_score = DCF(confmatrix, pi_true)
            dcfs[0].append(actDCF_score)
            minDCF_score = DCF_min(scores, val_labels, pi_true)
            dcfs[1].append(minDCF_score)
            # saving model parameters and scores
            np.save(f'results/svm/svm(C={C:.4f},K={K})_{to_snake_case(title)}_scores', scores)
            np.savez(f'models/svm/svm_(C={C:.4f},K={K})_{to_snake_case(title)}_params', 
                        weights=model.params[0], bias=model.params[1])
        plot_svm_dcfs(C_values, dcfs, f'SVM (K={K}), {title}', f'plots/svm_{to_snake_case(title)}_C_DCFs_plot')
    # analyze the DCF of the kernel SVM (polynomial kernel) as regularization changes
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    #train_data, train_labels = train_data[:, ::50], train_labels[::50] # just for experiments
    d, c = 2, 1
    K = 0
    plot_dict = {}
    for d, c in [(2, 1), (4, 1)]:
        dcfs = [[], []]
        for C in C_values:
            model = SVM(C, K, poly_kernel(d, c))
            model.train(train_data, train_labels)
            scores = model(val_data)
            predictions = bayes_pred_llr(scores, pi_true)
            confmatrix = confusion_matrix(predictions, val_labels)
            actDCF_score = DCF(confmatrix, pi_true)
            dcfs[0].append(actDCF_score)
            minDCF_score = DCF_min(scores, val_labels, pi_true)
            dcfs[1].append(minDCF_score)
            plot_dict[f'Pol(d={d:.3f}, c={c:.3f})'] = dcfs
            # saving the model and scores
            np.save(f'results/svm/svm(C={C:.4f},K={K})_poly(d={d},c={c})_scores', scores)
            np.savez(f'models/svm/svm_(C={C:.4f},K={K})_poly(d={d},c={c})_params', 
                        weights=model.params[0], bias=model.params[1])
    plot_kernsvm_dcfs(C_values, plot_dict, f'SVM (K={K})', 'plots/svm_poly_C_DCFs_plot')
    # analyze the DCF of the kernel SVM (RBF kernel) as regularization changes
    C = np.logspace(-3, 2, 11)
    g_values = np.exp(-np.arange(1, 5)[::-1])
    K = 1
    plot_dict = {}
    for g in g_values:
        dcfs = [[], []]
        for C in C_values:
            model = SVM(C, K, rbf_kernel(g))
            model.train(train_data, train_labels)
            scores = model(val_data)
            predictions = bayes_pred_llr(scores, pi_true)
            confmatrix = confusion_matrix(predictions, val_labels)
            actDCF, minDCF = DCF(confmatrix, pi_true), DCF_min(scores, val_labels, pi_true)
            dcfs[0].append(actDCF)
            dcfs[1].append(minDCF)
            plot_dict[f'RBF(γ={g:.3f})'] = dcfs
            # save model parameters and scores
            np.save(f'results/svm/svm(C={C:.4f},K={K})_RBF(γ={g:.3f})_scores', scores)
            np.savez(f'models/svm/svm(C={C:.4f},K={K})_RBF(γ={g:.3f})_params', 
                    weights=model.params[0], bias=model.params[1])
    plot_kernsvm_dcfs(C_values, plot_dict, f'SVM (K={K})', 'plots/svm_RBF_C_DCFs_plot')

if __name__ == '__main__':
    main()