from libs.model_evaluation import bayes_pred_llr, DCF, DCF_min, confusion_matrix, plot_bayer_error
from libs.model_evaluation import effective_priors
from libs.utils import load_data, split_2to1, print_table
import os
import itertools
import numpy as np

applications = [
    (0.5, 1, 1),
    (0.9, 1, 1),
    (0.1, 1, 1),
    (0.5, 1, 9),
    (0.5, 9, 1)
]

basepath = 'results/gaussian_models'
results_filepath = 'results/gaussian_models.txt'
effective_prior = lambda p, Cfn, Cfp: (p * Cfn) / ((p * Cfn) + ((1 - p) * Cfp))

def main():
    priors_eff = set()
    # represent the applications in term of effective priors
    for p, Cfn, Cfp in applications:
        p_eff = effective_prior(p, Cfn, Cfp)
        priors_eff.add(p_eff)
    # optimal bayes decisions for every application-MVG model variant
    data, labels = load_data('dataset/train.txt')
    val_labels = split_2to1(data, labels)[1][1]
    table_rows = []
    for p_eff in priors_eff:
        for filename in os.listdir(basepath):
            filepath = os.path.join(basepath, filename)
            llr_scores = np.load(filepath)
            predictions = bayes_pred_llr(llr_scores, p_eff)
            M = confusion_matrix(predictions, val_labels)
            dcf_score = DCF(M, p_eff)
            dcfmin_score = DCF_min(llr_scores, val_labels, p_eff)
            table_rows.append((
                f'{p_eff:.1f}' if f'{p_eff:.1f}' not in itertools.chain(*table_rows) else '',
                f'{' '.join(filename.removesuffix('_llr.npy').capitalize().split('_'))}', 
                dcf_score, 
                dcfmin_score,
                dcf_score - dcfmin_score,
                f'{((dcf_score - dcfmin_score) / dcfmin_score) * 100:.2f}%'
            ))
    print_table(table_rows, ['𝜋̃', 'Model', 'DCF', 'min DCF', 'ΔDCF', 'Calibration loss'], filepath=results_filepath)
    # from the table the best PCA configuration is no PCA
    # load the llr scores for the models of this configuration
    model_llr_scores = []
    for filename in os.listdir(basepath):
        if 'm=' not in filename:
            filepath = os.path.join(basepath, filename)
            llr_scores = np.load(filepath)
            model_name = f'{' '.join(filename.removesuffix('_llr.npy').capitalize().split('_'))}'
            model_llr_scores.append((model_name, llr_scores))
    # compare the models for various effective priors using a bayes error plot
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