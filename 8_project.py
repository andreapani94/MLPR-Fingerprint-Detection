from libs.utils import load_data, split_2to1, to_snake_case, print_table
from libs.gaussian_mixture_models import GMM, save_gmm
from libs.model_evaluation import bayes_pred_llr, confusion_matrix, DCF, DCF_min, plot_bayer_error, prepare_bayes_plot_data
from libs.logistic_regression import LogisticRegression
from libs.support_vector_machines import SVM
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
TRAIN_GMM = False
 
def main():
    data, labels = load_data('dataset/train.txt')
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    if TRAIN_GMM:
        #  clear results file
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
            for G0 in G_values:
                for G1 in G_values:
                    model0, model1 = GMM(n_components=G0, covtype=covtype), GMM(n_components=G1, covtype=covtype)
                    model0.train(train_data[:, train_labels==0], conv_threshold=1e-6, covbound=0.01)
                    model1.train(train_data[:, train_labels==1], conv_threshold=1e-6, covbound=0.01)
                    llr_scores = model1(val_data) - model0(val_data)
                    predictions = bayes_pred_llr(llr_scores, pi_true)
                    confmatrix = confusion_matrix(predictions, val_labels)
                    actDCF = DCF(confmatrix, pi_true)
                    dcfs[0].append(actDCF)
                    minDCF = DCF_min(llr_scores, val_labels, pi_true)
                    dcfs[1].append(minDCF)
                    table_rows.append((G0, G1, actDCF, minDCF, f'{((actDCF - minDCF) / minDCF) * 100:.2f}%'))
                    np.save(f'results/gmm/GMM(G0={G0},G1={G1})_{covtype}_llrs', llr_scores) 
                    save_gmm(model0.params, f'models/gmm/GMM(G={G0})_{covtype}_class0_params')
                    save_gmm(model1.params, f'models/gmm/GMM(G={G1})_{covtype}_class1_params')
            #plot_gmm_dcfs(G_values, dcfs, f'GMM {title}', f'plots/GMM_{covtype}_G_DCFs_plot')
            print_table(table_rows, ['G0', 'G1', 'actDCF', 'minDCF', 'Calibration loss'], filepath=filepath, title=f'GMM {title}')

    # compare best performing models
    model_scores_map = {}
    #   load best performing Logistic Regression model (actDCF=0.497, minDCF=0.244)
    logreg_scores = np.load('results/logistic_regression/quadratic_logreg(λ=3.1623e-02)_llrs.npy')
    model_scores_map['Quadratic Logistic Regression (λ=3.1623e-02)'] = logreg_scores
    #   load best performing SVM model (actDCF=0.424, minDCF=0.172)
    svm_scores = np.load('results/svm/svm(C=3.1623e+01,K=1)_RBF(γ=1.3534e-01)_scores.npy')
    model_scores_map['SVM, RBF (γ=1.35334e-01)'] = svm_scores
    #   load best performing GMM model (actDCF=0.152, minDCF=0.131)
    gmm_scores = np.load('results/gmm/GMM(G0=8,G1=32)_diag_llrs.npy')
    model_scores_map['GMM Diagonal (G0=8, G1=32)'] = gmm_scores
    #   check scores are consistent 
    predictions = bayes_pred_llr(svm_scores, pi_true)
    cm = confusion_matrix(predictions, val_labels)
    actDCF, minDCF = DCF(cm, pi_true), DCF_min(svm_scores, val_labels, pi_true)
    model_dcf_map, prior_logodds = prepare_bayes_plot_data(model_scores_map, val_labels)
    plot_bayer_error(
        model_dcf_map=model_dcf_map,
        prior_logodds=prior_logodds,
        title='Best models comparison (uncalibrated)',
        savepath='plots/best_models_DCF_comparison.png'
    )
    

if __name__ == '__main__':
    main()