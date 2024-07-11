import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tabulate import tabulate


def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def confusion_matrix(lpred, ltrue):
    labels = np.unique(ltrue)
    M = np.zeros((len(labels), len(labels)), dtype=np.int32)
    
    for i in range(len(lpred)):
        M[lpred[i], ltrue[i]] += 1

    return M

def bayes_pred_llr(llr: np.array, pi: float, Cfn: float=1, Cfp: float=1, t: float=None):
    if t == None:
        t = - np.log((pi * Cfn) / ((1 - pi) * Cfp))
    preds = np.where(llr > t, 1, 0).ravel()
    return preds

def DCF(M: np.ndarray, pi: float, Cfn: float=1, Cfp: float=1, norm=True):
    Pfn = M[0, 1] / (M[1, 1] + M[0, 1])
    Pfp = M[1, 0] / (M[0, 0] + M[1, 0])
    y = Pfn * Cfn * pi + Pfp * Cfp * (1 - pi)
    if norm == True:
        y /= min(pi * Cfn, (1 - pi) * Cfp)
    return y

def DCF_min(S, ltrue, pi: float, Cfn: float=1, Cfp: float=1, return_threshold=False):
    dcf_values = []
    sorter = np.argsort(S)
    S_sorted = S[sorter]
    ltrue_sorted = ltrue[sorter]
    thresholds = np.hstack([-np.inf, S_sorted, np.inf])

    # initialize the confusion matrix with the result of -np.inf
    # that is all samples are classified as negatives
    lpred = np.where(S_sorted > -np.inf, 1, 0).ravel()
    CM = confusion_matrix(lpred, ltrue_sorted)
    dcf_t = DCF(CM, pi, Cfn, Cfp, norm=True)
    dcf_values.append(dcf_t)

    # for every threshold change the confusion matrix
    # based on the label of the sample that corresponds to it
    for i in range(len(S_sorted)):
        CM[0, ltrue_sorted[i]] += 1
        CM[1, ltrue_sorted[i]] -= 1
        dcf_t = DCF(CM, pi, Cfn, Cfp, norm=True)
        dcf_values.append(dcf_t)

    # with threshold np.inf all the samples are classified as positives
    lpred = np.where(S_sorted > np.inf, 1, 0).ravel()
    CM = confusion_matrix(lpred, ltrue_sorted)
    dcf_t = DCF(CM, pi, Cfn, Cfp, norm=True)
    dcf_values.append(dcf_t)

    # find the minimum
    min_idx = np.argmin(dcf_values)
        
    if return_threshold:
        return dcf_values[min_idx], thresholds[min_idx]
    else:
        return dcf_values[min_idx]


def plot_ROC_curve(scores, ltrue):
    xplot, yplot = [], []
    thresholds = np.linspace(-100, 100, 1000)

    for t in thresholds:
        lpreds = [1 if s >= t else 0 for s in scores]
        M = confusion_matrix(lpreds, ltrue)
        Pfn = M[0, 1] / (M[1, 1] + M[0, 1])
        Pfp = M[1, 0] / (M[0, 0] + M[1, 0])
        Ptp = 1 - Pfn
        xplot.append(Pfp)
        yplot.append(Ptp)
    plt.figure()
    plt.plot(xplot, yplot, color='blue')
    plt.grid()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig('ROC_curve.png')

def plot_bayer_error(prior_logodds, model_dcf_map: dict[str, dict[str, list[float]]], 
                        title: str=None, savepath: str=None, separate_plots: bool=False):
    plt.figure()
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    if not separate_plots:
        for model in model_dcf_map:
            color = next(colors)
            linestyles = cycle(['-', '--', ':'])
            for dcf_type in model_dcf_map[model]:
                linestyle = next(linestyles)
                plt.plot(prior_logodds, model_dcf_map[model][dcf_type], label=f'{dcf_type} ({model})', 
                            linewidth=2, color=color, linestyle=linestyle)
        plt.ylim([0, 1.1])
        plt.xlim([min(prior_logodds), max(prior_logodds)])
        plt.legend()
        plt.xlabel('Prior log-odds')
        plt.ylabel('DCF')
        plt.title(title) if title else plt.title('Bayes Error')
    else:
        num_models = len(model_dcf_map)
        num_cols = 2
        num_rows = (num_models + num_cols - 1) // num_cols
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
        axes = axes.flatten() 
        for i, model in enumerate(model_dcf_map):
            color = next(colors)
            linestyles = cycle(['-', '--', ':'])
            ax = axes[i]
            ax.set_title(model)
            ax.set_ylim([0, 1.1])
            plt.xlim([min(prior_logodds), max(prior_logodds)])
            ax.set_xlabel('Prior log-odds')
            ax.set_ylabel('DCF')
            for dcf_type in model_dcf_map[model]:
                linestyle = next(linestyles)
                ax.plot(prior_logodds, model_dcf_map[model][dcf_type], label=f'{dcf_type} ({model})', 
                            linewidth=2, color=color, linestyle=linestyle)
            ax.legend()
        # Turn off any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        plt.suptitle(title) if title else plt.suptitle('Bayes Error')

    plt.tight_layout()
    if not savepath:
        plt.savefig('bayes_error.png')
    else:
        plt.savefig(savepath)

def prepare_bayes_plot_data(model_scores_map: dict, labels: list):
    effprior_logodds = np.linspace(-4, 4, 21)
    effpriors = 1 / (1 + np.exp(-effprior_logodds))
    model_dcf_map = {}
    for model_name, scores in model_scores_map.items():
        actDCFs, minDCFs = list(), list()
        for prior_eff in effpriors:
            predictions = bayes_pred_llr(scores, prior_eff)
            M = confusion_matrix(predictions, labels)
            dcf_score = DCF(M, prior_eff)
            actDCFs.append(dcf_score)
            mindcf_score = DCF_min(scores, labels, prior_eff)
            minDCFs.append(mindcf_score)
        model_dcf_map[model_name] = {'actDCF': actDCFs, 'minDCF': minDCFs}
    return model_dcf_map, effprior_logodds

def bayes_pred_post(P: np.ndarray, C: np.ndarray):
    Bscores = C @ P
    preds = np.argmin(Bscores, 0)
    return preds

def DCF_mul_fast(M: np.ndarray, C: np.ndarray, pi: np.ndarray, norm=True):
    K = M.shape[0]
    R = M / M.sum(0)
    y = 0

    for k in range(K):
        y += pi[k] * (R[:, k] @ C[:, k])
    
    if norm == True:
        y /= np.min(C @ pi)
    
    return y

def effective_priors(prior_logodds: np.ndarray):
    return 1 / (1 + np.exp(-prior_logodds))


def print_matrix(matrix):
    cols = np.arange(matrix.shape[1])
    print(tabulate(matrix, headers=cols, showindex='always', tablefmt='fancy_grid'))
    print()

def print_results(data: list[tuple], headers: list[str]):
    print(tabulate(data, headers=headers, floatfmt='.3f', tablefmt='grid'))
    print()

    

def main():
    # CONFUSION MATRICES
    #   Confusion matrices for the Iris Dataset classification problem
    D, l = load_iris()
    (Dtr, ltr), (Dval, lval) = split_2to1(D, l)
    mvg = MVG()
    mvg.fit(Dtr, ltr)
    Spost = mvg(Dval)
    preds = np.argmax(Spost, 0)
    cm = confusion_matrix(preds, lval)
    print('Iris Dataset:')
    print('MVG:')
    print_matrix(cm)
    mvg_tc = TiedCovarianceMVG()
    mvg_tc.fit(Dtr, ltr)
    Spost = mvg_tc(Dval)
    preds = np.argmax(Spost, 0)
    cm = confusion_matrix(preds, lval)
    print('Tied Covariance MVG:')
    print_matrix(cm)
    #   Confusion matrix for the Divina Commedia tercets classification problem
    Spost = np.load('data/commedia_ll.npy')
    labels = np.load('data/commedia_labels.npy')
    preds = np.argmax(Spost, 0)
    cm = confusion_matrix(preds, labels)
    print('Divina Commedia:')
    print_matrix(cm)

    # BINARY TASK: OPTIMAL DECISION, EVALUATION
    llr = np.load('data/commedia_llr_infpar.npy')
    ltrue = np.load('data/commedia_labels_infpar.npy')
    configs = [
        (0.5, 1, 1), (0.8, 1, 1), 
        (0.5, 10, 1), (0.8, 1, 10)
    ]
    print('Inferno(1) - Paradiso(0) binary task:\n')
    DCFu_data, DCF_data, DCFmin_data = [], [], []
    for config in configs:
        pi, Cfn, Cfp = config 
        lpred = bayes_pred_llr(llr, pi, Cfn, Cfp)
        cm = confusion_matrix(lpred, ltrue)
        print(f'(π={pi}, Cfn={Cfn}, Cfp={Cfp}):')
        print_matrix(cm)
        dcfu_score = DCF(cm, pi, Cfn, Cfp, norm=False)
        dcf_score = DCF(cm, pi, Cfn, Cfp)
        dcfmin_score = DCF_min(llr, ltrue, pi, Cfn, Cfp)
        DCFu_data.append((f'({pi}, {Cfn}, {Cfp})', dcfu_score))
        DCF_data.append((f'({pi}, {Cfn}, {Cfp})', dcf_score))
        DCFmin_data.append((f'({pi}, {Cfn}, {Cfp})', dcfmin_score))
    print_results(DCFu_data, ['(π, Cfn, Cfp)', 'DCF_u'])
    print_results(DCF_data, ['(π, Cfn, Cfp)', 'DCF'])
    print_results(DCFmin_data, ['(π, Cfn, Cfp)', 'min DCF'])
        
    #   ROC curves
    plot_ROC_curve(llr, labels)
    #   Bayer error plot
    llr = np.load('data/commedia_llr_infpar.npy')
    ltrue = np.load('data/commedia_labels_infpar.npy')
    pi, Cfn, Cfp = configs[0]
    lpred = bayes_pred_llr(llr, pi, Cfn, Cfp)
    M = confusion_matrix(lpred, ltrue)
    prior_logodds = np.linspace(-3, 3, 21)
    p_eff = lambda p: 1 / (1 + np.exp(-p)) # effective prior
    dcf = [DCF(M, p_eff(pi), 1, 1) for pi in prior_logodds]
    dcfmin = [DCF_min(llr, ltrue, p_eff(pi), 1, 1) for pi in prior_logodds]
    model_dcf_map = {'ε = 0.001': (dcf, dcfmin)}
    plot_bayer_error(prior_logodds, model_dcf_map)
    #       compare two classifiers
    llr = np.load('data/commedia_llr_infpar_eps1.npy')
    ltrue = np.load('data/commedia_labels_infpar_eps1.npy')
    lpred = bayes_pred_llr(llr, pi, Cfn, Cfp)
    M = confusion_matrix(lpred, ltrue)
    dcf = [DCF(M, p_eff(pi), 1, 1, norm=True) for pi in prior_logodds]
    dcfmin = [DCF_min(llr, ltrue, p_eff(pi), 1, 1) for pi in prior_logodds]
    model_dcf_map['ε = 1'] = (dcf, dcfmin)
    plot_bayer_error(prior_logodds, model_dcf_map, 'recognizer_comparison_dcf.png')

    # MULTICLASS TASK
    #   first with different priors than with uniform priors
    C_vals = [np.array([
        [0, 1, 2],
        [1, 0, 1],
        [2, 1, 0]
    ]), np.ones((3,3)) - np.eye(3)]
    pi_vals = [col(np.array([0.3, 0.4, 0.3])), np.full((3, 1), 1/3)]
    dcf_data = []
    for C, pi in zip(C_vals, pi_vals):
        for eps in [0.001, 1]:
            if eps == 0.001:
                ll = np.load('data/commedia_ll.npy')
                ltrue = np.load('data/commedia_labels.npy')
            else:
                ll = np.load('data/commedia_ll_eps1.npy')
                ltrue = np.load('data/commedia_labels_eps1.npy')
            Spost = np.exp(ll + np.log(pi))
            lpred = bayes_pred_post(Spost, C)
            cm = confusion_matrix(lpred, ltrue)
            print(f'ε={eps}:')
            print_matrix(cm)
            dcfu_score = DCF_mul_fast(cm, C, pi, norm=False)
            dcf_score = DCF_mul_fast(cm, C, pi)
            dcf_data.append((eps, dcfu_score, dcf_score))
        print_results(dcf_data, ['ε', 'DCFu', 'DCF'])
        dcf_data.clear()     

if __name__ == '__main__':
    from gaussian_models import load_iris, split_2to1, MVG, TiedCovarianceMVG
    main()