import numpy as np
from libs.logistic_regression import LogisticRegression
from libs.utils import load_data, split_2to1, row, col, print_table
from libs.model_calibration import Kfold_split
from libs.model_evaluation import bayes_pred_llr, confusion_matrix, DCF, DCF_min, plot_bayer_error

best_scores_filenames = [
    ('Quadratic Logistic Regression (λ=3.1623e-02)', 'results/logistic_regression/quadratic_logreg(λ=3.1623e-02)_llrs.npy'),
    ('Diagonal GMM (G=8)', 'results/gmm/GMM(G=8)_diag_llrs.npy')
]

pi_true = 0.1
K = 5
   
def main():
    data, labels = load_data('dataset/train.txt')
    _, (_, val_labels) = split_2to1(data, labels)
    model_scores_map = {}
    scores_all = []
    table_rows = []
    # score calibration
    for title, score_name in best_scores_filenames:
        scores = row(np.load(score_name))
        scores_all.append(scores)
        model_cal = LogisticRegression(prior=pi_true)
        Kfold_scores, Kfold_labels = list(), list()
        for (train_cal_data, train_cal_labels), (val_cal_data, val_cal_labels) in \
            Kfold_split(scores, val_labels, K=K):
                model_cal.train(train_cal_data, train_cal_labels)
                llr_scores_cal = model_cal(val_cal_data) - np.log(pi_true / (1 - pi_true))
                Kfold_scores.append(llr_scores_cal)
                Kfold_labels.append(val_cal_labels)
        # pool the scores and labels
        Kfold_scores, Kfold_labels = np.hstack(Kfold_scores), np.hstack(Kfold_labels)
        # compute actDCF and minDCF of the pooled calibrated scores
        predictions = bayes_pred_llr(Kfold_scores, pi_true)
        confmatrix = confusion_matrix(predictions, Kfold_labels)
        actDCF = DCF(confmatrix, pi_true)
        minDCF = DCF_min(Kfold_scores, Kfold_labels, pi_true)
        table_rows.append((title, actDCF, minDCF))
        model_scores_map[title] = ((Kfold_scores, Kfold_labels), (scores, val_labels))
    # score fusion
    scores_all = np.vstack(scores_all)
    Kfold_scores, Kfold_labels = list(), list()
    for (train_cal_data, train_cal_labels), (val_cal_data, val_cal_labels) in \
        Kfold_split(scores_all, val_labels, K=K):
            model_cal.train(train_cal_data, train_cal_labels)
            llr_scores_cal = model_cal(val_cal_data) - np.log(pi_true / (1 - pi_true))
            Kfold_scores.append(llr_scores_cal)
            Kfold_labels.append(val_cal_labels)
    #   pool the scores and labels
    Kfold_scores, Kfold_labels = np.hstack(Kfold_scores), np.hstack(Kfold_labels)
    # compute actDCF and minDCF of the fused scores
    predictions = bayes_pred_llr(Kfold_scores, pi_true)
    confmatrix = confusion_matrix(predictions, Kfold_labels)
    actDCF = DCF(confmatrix, pi_true)
    minDCF = DCF_min(Kfold_scores, Kfold_labels, pi_true)
    table_rows.append(('Model fusion', actDCF, minDCF))
    model_scores_map['Model fusion'] = ((Kfold_scores, Kfold_labels), (Kfold_scores, Kfold_labels))
    # model comparison using a bayes error plot
    effprior_logodds = np.linspace(-4, 4, 21)
    effpriors = 1 / (1 + np.exp(-effprior_logodds))
    model_dcf_map = {}
    for model_name in model_scores_map:
        actDCFs_cal, minDCFs, actDCFs = list(), list(), list()
        (scores_cal, labels_cal), (scores, labels) = model_scores_map[model_name]
        scores = scores.ravel()
        for prior_eff in effpriors:
            predictions = bayes_pred_llr(scores_cal, prior_eff)
            M = confusion_matrix(predictions, labels_cal)
            dcf_score = DCF(M, prior_eff)
            actDCFs_cal.append(dcf_score)
            predictions = bayes_pred_llr(scores, prior_eff)
            M = confusion_matrix(predictions, labels)
            dcf_score = DCF(M, prior_eff)
            actDCFs.append(dcf_score)
            mindcf_score = DCF_min(scores, labels, prior_eff)
            minDCFs.append(mindcf_score)
        model_dcf_map[model_name] = {'actDCF (calibrated)': actDCFs_cal, 'minDCF': minDCFs, 'actDCF': actDCFs}
    plot_bayer_error(
        prior_logodds=effprior_logodds,
        model_dcf_map=model_dcf_map,
        savepath='plots/best_models_DCF_comparison',
        title='Best models',
    )
    print_table(table_rows, ['', 'actDCF', 'minDCF'])



if __name__ == '__main__':
    main()