import numpy as np
from libs.logistic_regression import LogisticRegression, quadratic_expansion, empirical_prior_logodds
from libs.support_vector_machines import SVM, rbf_kernel
from libs.gaussian_mixture_models import GMM, load_gmm
from libs.utils import load_data, split_2to1, row, col, print_table
from libs.model_calibration import Kfold_split
from libs.model_evaluation import bayes_pred_llr, confusion_matrix, DCF, DCF_min
from libs.model_evaluation import plot_bayer_error, prepare_bayes_plot_data

# a dictionary containing scores, parameters and metrics of the best models
model_data_map = {
    'Quadratic Logistic Regression (λ=3.1623e-02)': {},
    'SVM, RBF (γ=1.35334e-01)': {},
    'GMM Diagonal (G0=8, G1=32)': {},
    'Model Fusion': {}
}

pi_true = 0.1
K = 5
filepath = 'results/evaluation.txt'

def main():
    data, labels = load_data('dataset/train.txt')
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    #  clear results file
    with open(filepath, 'w') as f:
        f.truncate()

    # CALIBRATION
    # load scores and model parameters of the best performing models
    #   load best performing Logistic Regression model (actDCF=0.497, minDCF=0.244)
    logreg_scores = np.load('results/logistic_regression/quadratic_logreg(λ=3.1623e-02)_llrs.npy')
    logreg = LogisticRegression()
    logreg.load('models/logistic_regression/quadratic_logreg(λ=3.1623e-02)_params.npz')
    model_data_map['Quadratic Logistic Regression (λ=3.1623e-02)']['scores'] = row(logreg_scores)
    model_data_map['Quadratic Logistic Regression (λ=3.1623e-02)']['model'] = logreg
    #   load best performing SVM model (actDCF=0.424, minDCF=0.172)
    svm_scores = np.load('results/svm/svm(C=3.1623e+01,K=1)_RBF(γ=1.3534e-01)_scores.npy')
    svm = SVM(kernel=rbf_kernel(0.13533528323661267))
    svm.load('models/svm/svm(C=3.1623e+01,K=1)_RBF(γ=1.3534e-01)_params.npz')
    model_data_map['SVM, RBF (γ=1.35334e-01)']['scores'] = row(svm_scores)
    model_data_map['SVM, RBF (γ=1.35334e-01)']['model'] = svm
    #   load best performing GMM model (actDCF=0.152, minDCF=0.131)
    gmm_eval_scores = np.load('results/gmm/GMM(G0=8,G1=32)_diag_llrs.npy')
    gmm0 = GMM(params_init=load_gmm('models/gmm/GMM(G=8)_diag_class0_params'))
    gmm1 = GMM(params_init=load_gmm('models/gmm/GMM(G=32)_diag_class1_params'))
    model_data_map['GMM Diagonal (G0=8, G1=32)']['scores'] = row(gmm_eval_scores)
    model_data_map['GMM Diagonal (G0=8, G1=32)']['model'] = (gmm0, gmm1)
    #   combine scores of the best performing models for model fusion
    fusion_scores = np.vstack([logreg_scores, svm_scores, gmm_eval_scores])
    model_data_map['Model Fusion']['scores'] = fusion_scores

    # using a K-fold approach, find the best performing calibration transformation for every model
    calib_train_priors = 1 / (1 + np.exp(-np.linspace(-4, 4, 21)))
    for model_name in model_data_map:
        scores = model_data_map[model_name]['scores']
        actDCF_prior_map = {} # to store every calibration model trained on different priors
        for train_prior in calib_train_priors:
            cal_model = LogisticRegression(prior=train_prior)
            pooled_Kfold_scores, pooled_Kfold_labels = [], []
            kfolds = Kfold_split(scores, val_labels, K=K, shuffle=True)
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
        cal_model.train(scores, val_labels)
        model_data_map[model_name]['calibration_model'] = cal_model
        
    # print results
    table_rows = []
    #   calibrated models
    for model_name in model_data_map:
        if model_name != 'Model Fusion':
            # pre-calibration metrics
            scores = model_data_map[model_name]['scores'].ravel()
            predictions = bayes_pred_llr(scores, pi_true)
            cm = confusion_matrix(predictions, val_labels)
            actDCF = DCF(cm, pi_true)
            minDCF = DCF_min(scores, val_labels, pi_true)
            # calibrated metrics
            cal_model = model_data_map[model_name]['calibration_model']
            train_prior = cal_model.pi
            cal_scores = cal_model(row(scores)) - np.log(train_prior / (1 - train_prior))
            model_data_map[model_name]['calibrated_scores'] = cal_scores
            predictions = bayes_pred_llr(cal_scores, pi_true)
            cm = confusion_matrix(predictions, val_labels)
            actDCF_cal = DCF(cm, pi_true)
            minDCF_cal = DCF_min(cal_scores, val_labels, pi_true)
            table_rows.append((model_name, actDCF, actDCF_cal, minDCF, minDCF_cal, train_prior))
    print_table(table_rows, ['', 'actDCF', 'actDCF (calibrated)', 'minDCF', 'minDCF (calibrated)', 'π (calibration)'],
                filepath=filepath, title='Best models (calibrated)')
    #   model fusion
    table_rows.clear()
    scores = model_data_map['Model Fusion']['scores']
    fusion_model = model_data_map['Model Fusion']['calibration_model']
    train_prior = fusion_model.pi
    fused_scores = fusion_model(scores) - np.log(train_prior / (1 - train_prior))
    model_data_map[model_name]['calibrated_scores'] = fused_scores
    predictions = bayes_pred_llr(fused_scores, pi_true)
    cm = confusion_matrix(predictions, val_labels)
    actDCF = DCF(cm, pi_true)
    minDCF = DCF_min(fused_scores, val_labels, pi_true)
    table_rows.append((model_name, actDCF, minDCF, train_prior))
    print_table(table_rows, ['', 'actDCF', 'minDCF', 'π (fusion)'], filepath=filepath)
    # test calibrated models (and fusion) for different applications with a bayes error plot
    model_dcf_map, prior_logodds = prepare_bayes_plot_data(
        { model_name: data['calibrated_scores'] for model_name, data in model_data_map.items() }, 
        val_labels
    )
    plot_bayer_error(
        model_dcf_map=model_dcf_map,
        prior_logodds=prior_logodds,
        title='Best models comparison (calibrated)',
        savepath='plots/best_models_calibrated_DCF_comparison.png'
    )


    # EVALUATION
    eval_data, eval_labels = load_data('dataset/eval.txt')
    table_rows.clear()
    # test the final delivered model on evaluation data
    stacked_eval_scores = np.vstack([
        logreg(quadratic_expansion(eval_data)) - empirical_prior_logodds(train_data, train_labels), 
        svm(eval_data), 
        gmm1(eval_data) - gmm0(eval_data)
    ])
    fusion_model = model_data_map['Model Fusion']['calibration_model']
    train_prior = fusion_model.pi
    fused_eval_scores = fusion_model(stacked_eval_scores) - np.log(train_prior / (1 - train_prior))
    model_data_map['Model Fusion']['eval_scores'] = fused_eval_scores
    predictions = bayes_pred_llr(fused_eval_scores, pi_true)
    cm = confusion_matrix(predictions, eval_labels)
    actDCF = DCF(cm, pi_true)
    minDCF = DCF_min(fused_eval_scores, eval_labels, pi_true)
    table_rows.append(('Model Fusion', actDCF, minDCF))
    model_dcf_map, prior_logodds = prepare_bayes_plot_data({'Model Fusion': fused_eval_scores}, eval_labels)
    plot_bayer_error(
        model_dcf_map=model_dcf_map,
        prior_logodds=prior_logodds,
        title='Model Fusion (evaluation data)',
        savepath='plots/fusion_model_evaluation_error_plot.png'
    )
    # test all the best performing models on evaluation data
    #   Logistic Regression
    logreg_eval_scores = logreg(quadratic_expansion(eval_data)) - empirical_prior_logodds(train_data, train_labels)
    model_cal = model_data_map['Quadratic Logistic Regression (λ=3.1623e-02)']['calibration_model']
    logreg_eval_scores_cal = model_cal(row(logreg_eval_scores)) - np.log(model_cal.pi / (1 - model_cal.pi))
    model_data_map['Quadratic Logistic Regression (λ=3.1623e-02)']['eval_scores'] = logreg_eval_scores_cal
    predictions = bayes_pred_llr(logreg_eval_scores_cal, pi_true)
    cm = confusion_matrix(predictions, eval_labels)
    actDCF = DCF(cm, pi_true)
    minDCF = DCF_min(logreg_eval_scores_cal, eval_labels, pi_true)
    table_rows.append(('Quadratic Logistic Regression (λ=3.1623e-02)', actDCF, minDCF))
    #   SVM
    svm_eval_scores = svm(eval_data)
    model_cal = model_data_map['SVM, RBF (γ=1.35334e-01)']['calibration_model']
    svm_eval_scores_cal = model_cal(row(svm_eval_scores)) - np.log(model_cal.pi / (1 - model_cal.pi))
    model_data_map['SVM, RBF (γ=1.35334e-01)']['eval_scores'] = svm_eval_scores_cal
    predictions = bayes_pred_llr(svm_eval_scores_cal, pi_true)
    cm = confusion_matrix(predictions, eval_labels)
    actDCF = DCF(cm, pi_true)
    minDCF = DCF_min(svm_eval_scores_cal, eval_labels, pi_true)
    table_rows.append(('SVM, RBF (γ=1.35334e-01)', actDCF, minDCF))    
    #   GMM
    gmm_eval_scores = gmm1(eval_data) - gmm0(eval_data)
    model_cal = model_data_map['GMM Diagonal (G0=8, G1=32)']['calibration_model']
    gmm_eval_scores_cal = model_cal(row(gmm_eval_scores)) - np.log(model_cal.pi / (1 - model_cal.pi))
    model_data_map['GMM Diagonal (G0=8, G1=32)']['eval_scores'] = gmm_eval_scores_cal
    predictions = bayes_pred_llr(gmm_eval_scores_cal, pi_true)
    cm = confusion_matrix(predictions, eval_labels)
    actDCF = DCF(cm, pi_true)
    minDCF = DCF_min(gmm_eval_scores_cal, eval_labels, pi_true)
    table_rows.append(('GMM Diagonal (G0=8, G1=32)', actDCF, minDCF))

    print_table(table_rows, ['', 'actDCF (evaluation)', 'minDCF (evaluation)'], filepath=filepath)
    model_scores_map = {model_name: data['eval_scores'] for model_name, data in model_data_map.items()}
    model_dcf_map, prior_logodds = prepare_bayes_plot_data(model_scores_map, eval_labels)
    plot_bayer_error(
        model_dcf_map=model_dcf_map,
        prior_logodds=prior_logodds,
        title='Best models (evaluation data)',
        savepath='plots/best_models_evaluation_error_plot.png'
    )

    # analyze wether another training strategy would have led to a better results on evaluation
    G_values = 2**np.arange(0, 6)
    for title, covtype in [
            ('Full Covariance', 'full'), 
            ('Diagonal Covariance', 'diag'), 
            ('Tied Covariance', 'tied')
        ]:
        table_rows = []
        for G0 in G_values:
            for G1 in G_values:
                gmm0 = GMM(params_init=load_gmm(f'models/gmm/GMM(G={G0})_{covtype}_class0_params'))
                gmm1 = GMM(params_init=load_gmm(f'models/gmm/GMM(G={G1})_{covtype}_class1_params'))
                gmm_eval_scores = gmm1(eval_data) - gmm0(eval_data)
                predictions = bayes_pred_llr(gmm_eval_scores, pi_true)
                cm = confusion_matrix(predictions, eval_labels)
                minDCF_eval = DCF_min(gmm_eval_scores, eval_labels, pi_true)
                gmm_eval_scores = gmm1(val_data) - gmm0(val_data)
                predictions = bayes_pred_llr(gmm_eval_scores, pi_true)
                cm = confusion_matrix(predictions, val_labels)
                minDCF = DCF_min(gmm_eval_scores, val_labels, pi_true)
                table_rows.append((G0, G1, minDCF_eval, minDCF, 
                                    f'{((minDCF_eval - minDCF) / minDCF) * 100:.2f}%'))
        print_table(table_rows, ['G0', 'G1', 'minDCF (evaluation)', 'minDCF (validation)', 'DCF loss'], 
                        title=f'GMM {title} (evaluation)', filepath=filepath)

if __name__ == '__main__':
    main()