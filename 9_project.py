import numpy as np
from libs.logistic_regression import LogisticRegression
from libs.utils import load_data, split_2to1, row, col
from libs.model_calibration import Kfold_split
from libs.model_evaluation import bayes_pred_llr, confusion_matrix, DCF, DCF_min

best_scores_filenames = [
    'results/logistic_regression/quadratic_logreg(λ=3.1623e-02)_llrs.npy',
    'results/gmm/GMM(G=8)_diag_llrs.npy'
]

pi_true = 0.1
K = 5
   
def main():
    data, labels = load_data('dataset/train.txt')
    _, (_, val_labels) = split_2to1(data, labels)
    calibrated_scores = []
    # load the scores of the best performing models
    for score_name in best_scores_filenames:
        scores = row(np.load(score_name))
        model_cal = LogisticRegression(prior=pi_true)
        Kfold_scores, Kfold_labels = list(), list()
        for (train_cal_data, train_cal_labels), (val_cal_data, val_cal_labels) in \
            Kfold_split(scores, val_labels, K=K, shuffle=True):
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
        pass
    effprior_logodds = np.linspace(-4, 4, 21)
    effpriors = 1 / (1 + np.exp(-effprior_logodds))
    model_dcf_map = {}
    for model_name, llr_scores in model_llr_scores:
        actDCFs, minDCFs = list(), list()
        for prior_eff in effpriors:
            predictions = bayes_pred_llr(llr_scores, prior_eff)
            M = confusion_matrix(predictions, val_labels)
            dcf_score = DCF(M, prior_eff)
            actDCFs.append(dcf_score)
            mindcf_score = DCF_min(llr_scores, val_labels, prior_eff)
            minDCFs.append(mindcf_score)
        model_dcf_map[model_name] = {'actDCF': actDCFs, 'minDCF': minDCFs}
    plot_bayer_error(
        prior_logodds=effprior_logodds,
        model_dcf_map=model_dcf_map,
        savepath='plots/gaussian_models_DCF_comparison',
        title='MVG Variants',
        separate_plots=False
    )



if __name__ == '__main__':
    main()