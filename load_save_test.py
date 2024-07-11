from libs.logistic_regression import LogisticRegression
from libs.support_vector_machines import SVM, rbf_kernel, poly_kernel
from libs.utils import row, col
import numpy as np

def main():
    model = LogisticRegression(lambda_reg=10)
    X_train, y_train = np.arange(20).reshape((4, 5)), np.ones(5)
    model.train(X_train, y_train)
    x_test = col(np.arange(4))
    score = model(x_test)
    print(model.params)
    print(score)
    model.save('logreg_test')
    model2 = LogisticRegression()
    model2.load('logreg_test.npz')
    score = model2(x_test)
    print(model2.params)
    print(score)

    model = SVM(C=10, K=10)
    model.train(X_train, y_train)
    score = model(x_test)
    print(model.params)
    print(score)
    model.save('svm_test')
    model2 = SVM()
    model2.load('svm_test.npz')
    score = model2(x_test)
    print(model2.params)
    print(score)

    model = SVM(C=10, K=10, kernel=poly_kernel(2, 6))
    func = rbf_kernel(0.5)
    model.train(X_train, y_train)
    score = model(x_test)
    print(model.params)
    print(score)
    model.save('svm_test')
    model2 = SVM(kernel=poly_kernel(2, 6))
    model2.load('svm_test.npz')
    score = model2(x_test)
    print(model2.params)
    print(score)

if __name__ == '__main__':
    main()