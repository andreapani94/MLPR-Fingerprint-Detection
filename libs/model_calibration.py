import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tabulate import tabulate

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def print_results(data: list[tuple], headers: list[str]):
    print(tabulate(data, headers=headers, floatfmt='.3f', tablefmt='grid'))
    print()

def plot_calibration_results(prior_logodds, model_dcf_map: dict[str: tuple], title: str=None):
    plt.figure()
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    num_models = len(model_dcf_map)
    num_cols = 2
    num_rows = (num_models + num_cols - 1) // num_cols
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5 * num_rows))
    axes = axes.flatten() 

    for i, model in enumerate(model_dcf_map):
        ax = axes[i]
        color = next(colors)
        ax.set_title(model)
        ax.plot(prior_logodds, model_dcf_map[model][0], label=f'actDCF (calibrated)', linewidth=2, color=color)
        ax.plot(prior_logodds, model_dcf_map[model][1], label=f'actDCF (pre-calibration)', linewidth=2, linestyle=':', color=color)
        ax.plot(prior_logodds, model_dcf_map[model][2], label=f'min DCF ({model})', linewidth=2, linestyle='--', color=color)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.set_xlim([-3, 3])

    # Turn off any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.ylabel('DCF')
    plt.xlabel('Prior log-odds')
    #plt.title('Bayes Error')
    plt.tight_layout()
    if not title:
        plt.savefig('bayes_error.png')
    else:
        plt.savefig(f'{title}.png')


def Kfold_split(data: np.ndarray, labels: np.ndarray, K: int=1, shuffle: bool=False)-> list[tuple[tuple]]:
    train_val_pairs = []

    if shuffle:
        idxs = np.random.permutation(data.shape[1])
        data = data[:, idxs]
        labels = labels[idxs]
    
    for i in range(K):
        train_val_pairs.append((
            (np.hstack([data[:, j::K] for j in range(K) if j != i]), 
                np.hstack([labels[j::K] for j in range(K) if j != i])),
            (data[:, i::K], labels[i::K])
        ))
    
    return train_val_pairs

