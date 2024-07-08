from libs.utils import load_data, split_2to1, col, row, to_snake_case
from libs.support_vector_machines import SVM, poly_kernel, rbf_kernel
from libs.model_evaluation import DCF, DCF_min, confusion_matrix, bayes_pred_llr
import numpy as np
import matplotlib.pyplot as plt

def plot_svm_dcfs(C_values: list[float], dcfs: list[list, list], title: str):
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(C_values, dcfs[0], label='actDCF')
    plt.plot(C_values, dcfs[1], linestyle='--', label='minDCF')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./plots/{title}')
    plt.close()

def plot_kernsvm_dcfs(C_values: list[float], param_dcfs: dict[str, list[list, list]], title: str):
    plt.figure()
    plt.xscale('log', base=10)
    
    colormap = plt.get_cmap('tab10')
    colors = colormap(np.linspace(0, 1, len(param_dcfs)))
    
    for color, param in zip(colors, param_dcfs):
        plt.plot(C_values, param_dcfs[param][0], color=color, label=f'actDCF ({param})')
        plt.plot(C_values, param_dcfs[param][1], color=color, linestyle='--', label=f'minDCF ({param})')
    
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./plots/{title}')
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
        (train_data, val_data, 'Full dataset'), 
        (train_data - mu, val_data - mu, 'Centered dataset') 
    ]:
        dcfs = [[], []]
        for C in C_values:
            model = SVM(C, K)
            model.train(train_data, train_labels)
            scores = model(val_data)
            predictions = np.where(scores > 0, 1, 0)
            M = confusion_matrix(predictions, val_labels)
            actDCF_score = DCF(M, pi_true)
            dcfs[0].append(actDCF_score)
            minDCF_score = DCF_min(scores, val_labels, pi_true)
            dcfs[1].append(minDCF_score)
            # saving the model and scores
            np.save(f'results/svm/svm(C={C}, K={K})_scores_{to_snake_case(title)}', scores)
            np.savez(f'models/svm_(C={C}, K={K})_{to_snake_case(title)} ', 
                        weights=model.params[0], bias=model.params[1])
        plot_svm_dcfs(C_values, dcfs, f'svm_C_DCFs_{to_snake_case(title)}')
    # analyze the DCF of the kernel SVM (polynomial kernel) as regularization changes
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    #train_data, train_labels = train_data[:, ::50], train_labels[::50] # just for experiments
    d, c = 2, 1
    K = 0
    dcfs = [[], []]
    for C in C_values:
        model = SVM(C, K, poly_kernel(d, c))
        model.train(train_data, train_labels)
        scores = model(val_data)
        predictions = np.where(scores > 0, 1, 0)
        M = confusion_matrix(predictions, val_labels)
        actDCF_score = DCF(M, pi_true)
        dcfs[0].append(actDCF_score)
        minDCF_score = DCF_min(scores, val_labels, pi_true)
        dcfs[1].append(minDCF_score)
        # saving the model and scores
        np.save(f'results/svm/svm(C={C}, K={K})_poly(d={d}, c={c})_scores', scores)
        np.savez(f'models/svm_(C={C}, K={K})_poly(d={d}, c={c})', 
                    weights=model.params[0], bias=model.params[1])
    plot_kernsvm_dcfs(C_values, {f'd={d:.3f},c={c:.3f}': dcfs}, f'svm_poly_C_DCFs')
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
            predictions = np.where(scores > 0, 1, 0)
            M = confusion_matrix(predictions, val_labels)
            actDCF, minDCF = DCF(M, pi_true), DCF_min(scores, val_labels, pi_true)
            dcfs[0].append(actDCF)
            dcfs[1].append(minDCF)
            plot_dict[f'γ={g:.3f}'] = dcfs
            # save model parameters and scores
            np.save(f'results/svm/svm(C={C}, K={K})_RBF(γ={d})_scores', scores)
            np.savez(f'models/svm(C={C}, K={K})_RBF(γ={d})', 
                    weights=model.params[0], bias=model.params[1])
    plot_kernsvm_dcfs(C_values, plot_dict, f'svm_RBF_C_DCFs')

if __name__ == '__main__':
    main()