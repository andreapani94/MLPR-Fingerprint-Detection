import numpy as np
import matplotlib.pyplot as plt
from libs.utils import load_data, split_2to1, col, row
from logistic_regression import LogisticRegression, print_logreg_res
from model_evaluation import DCF, DCF_min, confusion_matrix, bayes_pred_llr

pi = 0.1

def plot_logreg_dcfs(l_values: list[float], dcfs: list[list, list], title: str):
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(l_values, dcfs[0], label='actDCF')
    plt.plot(l_values, dcfs[1], linestyle='--', label='minDCF')
    plt.xlabel('λ (regularization parameter)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.savefig(f'./plots/{title}')
    plt.close()

def quadratic_expansion(data):
    data_exp = []

    for i in range(data.shape[1]):
        x = data[:, i:i+1]
        x_exp = np.vstack([col((x @ x.T).ravel()), x])
        data_exp.append(x_exp)

    return np.hstack(data_exp)
    

def main():
    D, l = load_data('dataset/train.txt')
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    # DCFs as λ changes analysis 
    # once with a full dataset once with only 1/50 of the samples
    lam_values = np.logspace(-4, 2, 13)
    for Dtr, ltr, title in [(Dtr, ltr, 'full'), (Dtr[:, ::50], ltr[::50], '1outof50')]:
        dcfs = [[], []]
        for lam in lam_values:
            model = LogisticRegression(lam)
            # training
            model.train(Dtr, ltr)
            # evaluation
            S = model(Dval)
            lpred = np.where(S > 0, 1, 0).ravel()
            pi_emp = Dtr[:, ltr==1].shape[1] / Dtr.shape[1]
            Sllr = S - np.log(pi_emp / (1 - pi_emp))
            lpred = bayes_pred_llr(Sllr, pi)
            M = confusion_matrix(lpred, lval)
            actDCF = DCF(M, pi)
            minDCF = DCF_min(Sllr, lval, pi)
            dcfs[0].append(actDCF)
            dcfs[1].append(minDCF)
        plot_logreg_dcfs(lam_values, dcfs, f'logreg_λ_DCFs_{title}')
    # analyse the Prior-Weighted Logistic Regression
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    dcfs = [[], []]
    for lam in lam_values:
        model = LogisticRegression(lam, pi=pi)
        # training
        model.train(Dtr, ltr)
        # evaluation
        S = model(Dval)
        lpred = np.where(S > 0, 1, 0).ravel()
        Sllr = S - np.log(pi / (1 - pi))
        lpred = bayes_pred_llr(Sllr, pi)
        M = confusion_matrix(lpred, lval)
        actDCF = DCF(M, pi)
        minDCF = DCF_min(Sllr, lval, pi)
        dcfs[0].append(actDCF)
        dcfs[1].append(minDCF)
    plot_logreg_dcfs(lam_values, dcfs, f'prior_weighted_logreg_λ_DCFs_full')
    # analyze the quadratic Logistic Regression
    #   transform the features
    Dtr_exp, Dval_exp = quadratic_expansion(Dtr), quadratic_expansion(Dval)
    dcfs = [[], []]
    for lam in lam_values:
        model = LogisticRegression(lam)
        # training
        model.train(Dtr_exp, ltr)
        # evaluation
        S = model(Dval_exp)
        lpred = np.where(S > 0, 1, 0).ravel()
        pi_emp = Dtr_exp[:, ltr==1].shape[1] / Dtr_exp.shape[1]
        Sllr = S - np.log(pi_emp / (1 - pi_emp))
        lpred = bayes_pred_llr(Sllr, pi)
        M = confusion_matrix(lpred, lval)
        actDCF = DCF(M, pi)
        minDCF = DCF_min(Sllr, lval, pi)
        dcfs[0].append(actDCF)
        dcfs[1].append(minDCF)
    plot_logreg_dcfs(lam_values, dcfs, f'quad_logreg_λ_DCFs_full')
    # analyze the effect of centering on the Logistic Regression
    # can extend to other pre-processing strategies
    mu = col(np.mean(Dtr, 1))
    Dtr_c, Dval_c = Dtr - mu, Dval - mu
    dcfs = [[], []]
    for lam in lam_values:
        model = LogisticRegression(lam)
        # training
        model.train(Dtr_c, ltr)
        # evaluation
        S = model(Dval_c)
        lpred = np.where(S > 0, 1, 0).ravel()
        pi_emp = Dtr[:, ltr==1].shape[1] / Dtr.shape[1]
        Sllr = S - np.log(pi_emp / (1 - pi_emp))
        lpred = bayes_pred_llr(Sllr, pi)
        M = confusion_matrix(lpred, lval)
        actDCF = DCF(M, pi)
        minDCF = DCF_min(Sllr, lval, pi)
        dcfs[0].append(actDCF)
        dcfs[1].append(minDCF)
    plot_logreg_dcfs(lam_values, dcfs, f'logreg_λ_DCFs_full_centered')

if __name__ == '__main__':
    main()