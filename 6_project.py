from libs.utils import load_data, split_2to1, to_snake_case, zero_center, z_normalize, print_table
from libs.logistic_regression import LogisticRegression, empirical_prior_logodds, quadratic_expansion
from libs.model_evaluation import DCF, DCF_min, bayes_pred_llr, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

pi_true = 0.1
filepath = 'results/logistic_regression.txt'

def plot_logreg_dcfs(reg_values: list[float], dcfs: list[list, list], title: str, savepath: str):
    plt.figure()
    plt.xscale('log', base=10)
    plt.plot(reg_values, dcfs[0], label='actDCF', marker='o')
    plt.plot(reg_values, dcfs[1], linestyle='--', label='minDCF', marker='o')
    plt.xlabel('λ (regularization strength)')
    plt.ylabel('DCFs')
    plt.ylim(bottom=0)
    plt.legend()
    plt.title(title)
    plt.savefig(savepath)
    plt.close()


def main():
    data, labels = load_data('dataset/train.txt')
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    #   clear results file
    with open(filepath, 'w') as f:
        f.truncate()
    # analyze the DCF of the Logistic Regression as regularization changes
    #   once with a full dataset once with only 1/50 of the samples
    reg_values = np.logspace(-4, 2, 13)
    for i, (train_data, train_labels, title) in enumerate([
        (train_data, train_labels, 'Full dataset'),
        (train_data[:, ::50], train_labels[::50], '1 out of 50 dataset')
    ]):
        dcfs = [[], []]
        table_rows = []
        for lambda_reg in reg_values:
            model = LogisticRegression(lambda_reg=lambda_reg)
            # training
            model.train(train_data, train_labels)
            # classification
            scores = model(val_data)
            llr_scores = scores - empirical_prior_logodds(train_data, train_labels)
            predictions = bayes_pred_llr(llr_scores, pi_true)
            M = confusion_matrix(predictions, val_labels)
            actDCF_score = DCF(M, pi_true)
            dcfs[0].append(actDCF_score)
            minDCF_score = DCF_min(llr_scores, val_labels, pi_true)
            dcfs[1].append(minDCF_score)
            table_rows.append((
                f'{lambda_reg}', 
                actDCF_score, 
                minDCF_score, 
                f'{((actDCF_score - minDCF_score) / minDCF_score) * 100:.2f}%'
            ))
            # save model parameters and scores only for the full dataset
            if i == 0:
                np.save(f'results/logistic_regression/logreg(λ={lambda_reg:.4e})_llrs', llr_scores)
                np.savez(f'models/logistic_regression/logreg(λ={lambda_reg:.4e})_params', 
                            weights=model.params[0], bias=model.params[1], lambda_reg=lambda_reg)
        plot_logreg_dcfs(reg_values, dcfs, 'Logistic Regression', 
                            f'plots/logreg_λ_DCFs_{to_snake_case(title)}')
        print_table(table_rows, ['λ', 'actDCF', 'minDCF', 'Calibration loss'], 
                        title=f'Logistic Regression {title}', filepath=filepath)
    # analyse the DCF of the Prior-Weighted Logistic Regression as regularization changes
    (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
    dcfs = [[], []]
    table_rows.clear()
    for lambda_reg in reg_values:
        model = LogisticRegression(lambda_reg=lambda_reg, prior=pi_true)
        # training
        model.train(train_data, train_labels)
        # classification
        scores = model(val_data)
        llr_scores = scores - np.log(pi_true / (1 - pi_true))
        predictions = bayes_pred_llr(llr_scores, pi_true)
        M = confusion_matrix(predictions, val_labels)
        actDCF_score = DCF(M, pi_true)
        dcfs[0].append(actDCF_score)
        minDCF_score = DCF_min(llr_scores, val_labels, pi_true)
        dcfs[1].append(minDCF_score)
        table_rows.append((
            f'{lambda_reg}', 
            actDCF_score, 
            minDCF_score, 
            f'{((actDCF_score - minDCF_score) / minDCF_score) * 100:.2f}%'
        ))
        # save model parameters and scores
        np.save(f'results/logistic_regression/prior_weighted_logreg(λ={lambda_reg:.4e})_llrs', llr_scores)
        np.savez(f'models/logistic_regression/prior_weighted_logreg(λ={lambda_reg:.4e})_params', 
                    weights=model.params[0], bias=model.params[1], lambda_reg=lambda_reg)
    plot_logreg_dcfs(reg_values, dcfs, f'Logistic Regression (π={pi_true})', 
                        f'plots/prior_weighted_logreg_λ_DCFs_{to_snake_case('Full dataset')}')
    print_table(table_rows, ['λ', 'actDCF', 'minDCF', 'Calibration loss'], 
                    title='Prior-Weighted Logistic Regression', filepath=filepath)
    # analyse the DCF of the Quadratic Logistic Regression as regularization changes
    train_data_quad, val_data_quad = quadratic_expansion(train_data), quadratic_expansion(val_data)
    dcfs = [[], []]
    table_rows.clear()
    for lambda_reg in reg_values:
        model = LogisticRegression(lambda_reg=lambda_reg)
        # training
        model.train(train_data_quad, train_labels)
        # classification
        scores = model(val_data_quad)
        llr_scores = scores - empirical_prior_logodds(train_data_quad, train_labels)
        predictions = bayes_pred_llr(llr_scores, pi_true)
        M = confusion_matrix(predictions, val_labels)
        actDCF_score = DCF(M, pi_true)
        dcfs[0].append(actDCF_score)
        minDCF_score = DCF_min(llr_scores, val_labels, pi_true)
        dcfs[1].append(minDCF_score)
        table_rows.append((
            f'{lambda_reg}', 
            actDCF_score, 
            minDCF_score, 
            f'{((actDCF_score - minDCF_score) / minDCF_score) * 100:.2f}%'
        ))
        np.save(f'results/logistic_regression/quadratic_logreg(λ={lambda_reg:.4e})_llrs', llr_scores)
        np.savez(f'models/logistic_regression/quadratic_logreg(λ={lambda_reg:.4e})_params', 
                    weights=model.params[0], bias=model.params[1], lambda_reg=lambda_reg)
    plot_logreg_dcfs(reg_values, dcfs, 'Quadratic Logistic Regression', 
                        f'plots/quad_logreg_λ_DCFs_{to_snake_case('Full dataset')}')
    print_table(table_rows, ['λ', 'actDCF', 'minDCF', 'Calibration loss'], 
                    title='Quadratic Logistic Regression', filepath=filepath)
    # analyse the DCF of the Logistic Regression as regularization changes
    # with pre-processing techniques applied
    for title, preprocess in [
        ('Centered dataset', zero_center), 
        ('Z-normalized dataset', z_normalize)
    ]:
        (train_data, train_labels), (val_data, val_labels) = split_2to1(data, labels)
        train_data, val_data = preprocess(train_data, val_data)
        dcfs = [[], []]
        for lambda_reg in reg_values:
            model = LogisticRegression(lambda_reg=lambda_reg)
            # training
            model.train(train_data, train_labels)
            # classification
            scores = model(val_data)
            llr_scores = scores - empirical_prior_logodds(train_data, train_labels)
            predictions = bayes_pred_llr(llr_scores, pi_true)
            M = confusion_matrix(predictions, val_labels)
            actDCF_score = DCF(M, pi_true)
            dcfs[0].append(actDCF_score)
            minDCF_score = DCF_min(llr_scores, val_labels, pi_true)
            dcfs[1].append(minDCF_score)
        plot_logreg_dcfs(reg_values, dcfs, 'Logistic Regression', 
                            f'plots/logreg_λ_DCFs_{to_snake_case(title)}')


if __name__ == '__main__':
    main()