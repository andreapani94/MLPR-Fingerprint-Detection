import numpy as np
from libs.logistic_regression import LogisticRegression, quadratic_expansion, empirical_prior_logodds
from libs.support_vector_machines import SVM, rbf_kernel
from libs.gaussian_mixture_models import GMM, load_gmm
from libs.utils import load_data, split_2to1, row, col, print_table
from libs.model_calibration import Kfold_split
from libs.model_evaluation import bayes_pred_llr, confusion_matrix, DCF, DCF_min
from libs.model_evaluation import plot_bayer_error, prepare_bayes_plot_data

best_models_filenames = [
    ('Quadratic Logistic Regression (λ=3.1623e-02)', 'logistic_regression/quadratic_logreg(λ=3.1623e-02)'),
    ('Diagonal GMM (G=8)', 'gmm/GMM(G=8)_diag')
] # to be transformed into a dict

# a dictionary containing scores, parameters and metrics of the best models
model_data_map = {
    'Quadratic Logistic Regression (λ=3.1623e-02)': {},
    'SVM, RBF (γ=1.35334e-01)': {},
    'GMM Diagonal (G0=8, G1=32)': {}
}

pi_true = 0.1
K = 5

def main2():
    data, labels = load_data('dataset/train.txt')
    _, (val_data, val_labels) = split_2to1(data, labels)

    # load scores and model parameters of the best performing models
    #   load best performing Logistic Regression model (actDCF=0.497, minDCF=0.244)
    logreg_scores = np.load('results/logistic_regression/quadratic_logreg(λ=3.1623e-02)_llrs.npy')
    logreg = LogisticRegression()
    logreg.load('models/logistic_regression/quadratic_logreg(λ=3.1623e-02)_params.npz')
    model_data_map['Quadratic Logistic Regression (λ=3.1623e-02)']['scores'] = logreg_scores
    model_data_map['Quadratic Logistic Regression (λ=3.1623e-02)']['model'] = logreg
    #   load best performing SVM model (actDCF=0.424, minDCF=0.172)
    svm_scores = np.load('results/svm/svm(C=3.1623e+01,K=1)_RBF(γ=1.3534e-01)_scores.npy')
    svm = SVM(rbf_kernel(0.13533528323661267))
    svm.load('models/svm/svm(C=3.1623e+01,K=1)_RBF(γ=1.3534e-01)_params.npz')
    model_data_map['SVM, RBF (γ=1.35334e-01)']['scores'] = svm_scores
    model_data_map['SVM, RBF (γ=1.35334e-01)']['model'] = svm
    #   load best performing GMM model (actDCF=0.152, minDCF=0.131)
    gmm_scores = np.load('results/gmm/GMM(G0=8,G1=32)_diag_llrs.npy')
    gmm0 = GMM(params_init=load_gmm('models/gmm/GMM(G=8)_diag_class0_params'))
    gmm1 = GMM(params_init=load_gmm('models/gmm/GMM(G=32)_diag_class1_params'))
    model_data_map['GMM Diagonal (G0=8, G1=32)']['scores'] = gmm_scores
    model_data_map['GMM Diagonal (G0=8, G1=32)']['model'] = (gmm0, gmm1)

    # using a K-fold approach, find the best performing calibration transformation for every model
    calib_train_priors = 1 / (1 + np.exp(-np.linspace(-4, 4, 21)))
    for model_name in model_data_map:
        scores = model_data_map[model_name]['scores']
        actDCF_prior_map = {} # to store every calibration model trained on different priors
        for train_prior in calib_train_priors:
            cal_model = LogisticRegression(prior=train_prior)
            pooled_Kfold_scores, pooled_Kfold_labels = [], []
            kfolds = Kfold_split(row(scores), val_labels, K=K, shuffle=True)
            for (train_cal_data, train_cal_labels), (val_cal_data, val_cal_labels) in kfolds:
                # train the calibration model using the training prior
                cal_model.train(train_cal_data, train_cal_labels)
                cal_llr_scores = cal_model(val_cal_data) - np.log(train_prior / (1 - train_prior))
                pooled_Kfold_scores.append(cal_llr_scores)
                pooled_Kfold_labels.append(val_cal_labels)
            pooled_Kfold_scores = np.hstack(pooled_Kfold_scores)
            pooled_Kfold_labels = np.hstack(pooled_Kfold_labels)
            # evaluate metrics on pooled scores
            predictions = bayes_pred_llr(pooled_Kfold_scores, pi_true)
            cm = confusion_matrix(predictions, pooled_Kfold_labels)
            actDCF_cal = DCF(cm, pi_true)
            minDCF_cal = DCF_min(pooled_Kfold_scores, pooled_Kfold_labels, pi_true)
            actDCF_prior_map[actDCF_cal] = train_prior
        # find the best calibration model
        actDCF_cal_min = min(actDCF_prior_map)
        train_prior_best = actDCF_prior_map[actDCF_cal_min]
        #   train a final calibration model using the best train prior found
        scores = model_data_map[model_name]['scores']
        cal_model = LogisticRegression(prior=train_prior_best)
        cal_model.train(row(scores), val_labels)
        model_data_map[model_name]['calibration_model'] = cal_model
        
    # print results
    table_rows = []
    for model_name in model_data_map:
        # pre-calibration metrics
        model = model_data_map[model_name]['model']
        scores = model_data_map[model_name]['scores']
        predictions = bayes_pred_llr(scores, pi_true)
        cm = confusion_matrix(predictions, val_labels)
        actDCF = DCF(cm, pi_true)
        minDCF = DCF_min(scores, val_labels, pi_true)
        # calibrated metrics
        cal_model = model_data_map[model_name]['calibration_model']
        train_prior = cal_model.pi
        cal_scores = cal_model(row(scores)) - np.log(train_prior / (1 - train_prior))
        model_data_map['calibrated_scores'] = cal_scores
        predictions = bayes_pred_llr(cal_scores, pi_true)
        cm = confusion_matrix(predictions, val_labels)
        actDCF_cal = DCF(cm, pi_true)
        minDCF_cal = DCF_min(cal_scores, val_labels, pi_true)
        table_rows.append((model_name, actDCF, actDCF_cal, minDCF, minDCF_cal, train_prior))
    print_table(table_rows, ['', 'actDCF', 'actDCF (calibrated)', 'minDCF', 'minDCF (calibrated)', 'π (calibration)'])
    # test calibrated models for different applications with a bayes error plot
    model_dcf_map, prior_logodds = prepare_bayes_plot_data(, val_labels)
    plot_bayer_error(
        model_dcf_map=model_dcf_map,
        prior_logodds=prior_logodds,
        title='Best models comparison (uncalibrated)',
        savepath='plots/best_models_DCF_comparison.png'
    )

                 
             
         
   
def main():
    data, labels = load_data('dataset/train.txt')
    _, (_, val_labels) = split_2to1(data, labels)
    model_scores_map = {}
    scores_all = []
    table_rows = []
    models_cal = []
    # score calibration
    for title, model_name in best_models_filenames:
        scores = row(np.load(f'results/{model_name}_llrs.npy'))
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
        models_cal.append(model_cal)
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
    
    # Evaluation
    eval_data, eval_labels = load_data('dataset/eval.txt')
    #   load model parameters and hyperparameters
    best_model_params = np.load(f'models/{best_models_filenames[0][1]}_params.npz')
    best_model = LogisticRegression(lambda_reg=best_model_params['lambda_reg'].item())
    best_model.params = best_model_params['weights'], best_model_params['bias'].item()
    #   evualuate the model
    scores = best_model(quadratic_expansion(eval_data)) - empirical_prior_logodds(eval_data, eval_labels)
    predictions = bayes_pred_llr(scores, pi_true)
    confmatrix = confusion_matrix(predictions, eval_labels)
    actDCF = DCF(confmatrix, pi_true)
    minDCF = DCF_min(scores, eval_labels, pi_true)
    model_dcf_map, prior_logodds = prepare_bayes_plot_data(
        { best_models_filenames[0][0]: scores }, 
        eval_labels
    )
    plot_bayer_error(
        model_dcf_map=model_dcf_map,
        prior_logodds=prior_logodds,
        title='Delivered model (Evaluation data)',
        savepath='plots/delivered_model_bayes_error'
    )

    



if __name__ == '__main__':
    main2()