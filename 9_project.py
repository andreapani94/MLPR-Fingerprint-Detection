import numpy as np
import matplotlib.pyplot as plt
from libs.utils import load_data, split_2to1, col, row
from support_vector_machines import SVM, poly_kernel, rbf_kernel
from model_evaluation import DCF, DCF_min, bayes_pred_llr, confusion_matrix

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
    for param in param_dcfs:
        plt.plot(C_values, param_dcfs[param][0], label=f'actDCF ({param})')
        plt.plot(C_values, param_dcfs[param][1], linestyle='--', label=f'minDCF ({param})')
    plt.xlabel('C (regularization parameter)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./plots/{title}')
    plt.close()

pi = 0.1

def main():
    D, l = load_data('dataset/train.txt')
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    C_values = np.logspace(-5, 0, 11)
    Dtr, ltr = Dtr[:, ::100], ltr[::100] # just for experiments
    # LINEAR SVM
    K = 1
    mu = col(np.mean(Dtr, 1))
    for Dtr, ltr, Dval, title in [(Dtr, ltr, Dval, ''), (Dtr - mu, ltr, Dval - mu, '_centered')]:
        dcfs = [[], []]
        for C in C_values:
            model = SVM(C, K)
            model.train(Dtr, ltr)
            S = model(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            err = 1 - (np.sum(lpred == lval) / len(lval))
            M = confusion_matrix(lpred, lval)
            actDCF, minDCF = DCF(M, pi), DCF_min(S, lval, pi)
            dcfs[0].append(actDCF)
            dcfs[1].append(minDCF)
            # save model parameters and scores
            np.save(f'./models/svm_C={C}', np.hstack([model.params[0].ravel(), model.params[1].ravel()]))
            np.save(f'./results/svm_C={C}_scores', S)
        plot_svm_dcfs(C_values, dcfs, f'svm_C_DCFs{title}')

    # KERNEL SVM
    #   polynomial kernel
    d, c = 2, 1
    K = 0
    dcfs = [[], []]
    for C in C_values:
            model = SVM(C, K, poly_kernel(d, c))
            model.train(Dtr, ltr)
            S = model(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            err = 1 - (np.sum(lpred == lval) / len(lval))
            M = confusion_matrix(lpred, lval)
            actDCF, minDCF = DCF(M, pi), DCF_min(S, lval, pi)
            dcfs[0].append(actDCF)
            dcfs[1].append(minDCF)
            # save model parameters and scores
            np.save(f'./models/svm_C={C}_poly(d={c},c={c})', np.hstack([model.params[0], model.params[1]]))
            np.save(f'./results/svm_C={C}_poly(d={c},c={c})_scores', S)
    plot_kernsvm_dcfs(C_values, {f'd={c},c={c}': dcfs}, f'kernsvm_C_DCFs_poly')
    #   RBF kernel
    C = np.logspace(-3, 2, 11)
    g_values = np.exp(-np.arange(1, 5, -1))
    K = 1
    plot_dict = {}
    for g in g_values:
        dcfs = [[], []]
        for C in C_values:
            model = SVM(C, K, rbf_kernel(g))
            model.train(Dtr, ltr)
            S = model(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            err = 1 - (np.sum(lpred == lval) / len(lval))
            M = confusion_matrix(lpred, lval)
            actDCF, minDCF = DCF(M, pi), DCF_min(S, lval, pi)
            dcfs[0].append(actDCF)
            dcfs[1].append(minDCF)
            plot_dict[f'γ={g}'] = dcfs
            # save model parameters and scores
            np.save(f'./models/svm_C={C}_RBF(γ={g}))', np.hstack([model.params[0], model.params[1]]))
            np.save(f'./results/svm_C={C}_RBF(γ={g}))_scores', S)
    plot_kernsvm_dcfs(C_values, plot_dict, f'kernsvm_C_DCFs_RBF')    


if __name__ == '__main__':
    main()