import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
import re

import matplotlib.pyplot as plt

label_names = {
    0: 'Counterfeit',
    1: 'Genuine'
}

def col(v: np.ndarray):
    return v.reshape(v.size, 1)

def row(v: np.ndarray):
    return v.reshape(1, v.size)

def load_data(path: str) -> np.ndarray:
    samples = []
    labels = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            fields = line.strip().split(',')
            sample = col(np.array(fields[:-1], np.float32))
            samples.append(sample)
            labels.append(int(fields[-1]))
    return np.hstack(samples), np.array(labels)

def split_2to1(D, l, seed=0):
    ntrain = int(D.shape[1] * 2 / 3)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxtrain = idx[:ntrain]
    idxval = idx[ntrain:]

    Dtr = D[:, idxtrain]
    Dval = D[:, idxval]
    ltr = l[idxtrain]
    lval = l[idxval]

    return (Dtr, ltr), (Dval, lval)

def plot_feature_hist(data: np.ndarray, labels: np.ndarray, path: str, title: str=None):
    for i in range(data.shape[0]):
        plt.figure()
        for label in np.unique(labels):
            mask = (labels == label)
            plt.hist(data[i, mask], bins=10, density=True, 
                     alpha=0.4, label=label_names[label], ec='black')
            plt.xlabel(f'Feature {i+1}')
            plt.legend()
            plt.tight_layout()
        if not title:
            plt.savefig(f'{path}/feature{i+1}_hist.png')
        else:
            plt.savefig(f'{path}/{title}_feature{i+1}_hist.png')
        plt.close()

def plot_feature_scatter(data: np.ndarray, labels: np.ndarray, path: str):
    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            plt.figure()
            if i != j:
                for label in np.unique(labels):
                    mask = (labels == label)
                    plt.scatter(data[i, mask], data[j, mask], label=label_names[label])
                    plt.xlabel(f'Feature {i+1}')
                    plt.ylabel(f'Feature {j+1}')
                    plt.legend()
                    plt.tight_layout()
                plt.savefig(f'{path}/feature{i+1}-{j+1}_scatter.png')
                plt.close()

def print_matrix(matrix, title: str=None, filepath: str=None):
    cols = [f"{i}" for i in range(len(matrix[0]))]
    try:
        if filepath:
            with open(filepath, 'a', encoding='utf-8') as f:
               if title: print(title, file=f)
               print(tabulate(matrix, headers=cols, showindex='always', tablefmt='fancy_grid', floatfmt='.3f'), file=f)
               print(file=f)
        else:
            if title: print(title)
            print(tabulate(matrix, headers=cols, showindex='always', tablefmt='fancy_grid', floatfmt='.3f'))
            print() 
    except OSError as error:
       print(error) 

def print_table(data: list[tuple], headers: list[str], title:str=None, filepath: str=None):
    try:
        if filepath:
            with open(filepath, 'a', encoding='utf-8') as f:
                if title: print(title, file=f)
                print(tabulate(data, headers=headers, floatfmt='.3f', tablefmt='grid'), file=f)
                print(file=f)    
        else:
            if title: print(title)
            print(tabulate(data, headers=headers, floatfmt='.3f', tablefmt='grid'))
            print()
    except OSError as error:
       print(error) 

def to_snake_case(title):
    title = title.lower()
    title = re.sub(r'\s+', '_', title)
    title = re.sub(r'[^a-z0-9_]', '', title)
    return title

def z_normalize(X_train: np.ndarray, X_val: np.ndarray):
    """
        Transforms both X_train and X_val using information
        (mean and standard deviation) from X_train
    """
    mu = col(np.mean(X_train, 1))
    var = np.std(X_train)
    return (X_train - mu) / var, (X_val - mu) / var

def zero_center(X_train: np.ndarray, X_val: np.ndarray):
    """
        Transforms both X_train and X_val using information
        (mean) from X_train
    """
    mu = col(np.mean(X_train, 1))
    return (X_train - mu), (X_val - mu)

