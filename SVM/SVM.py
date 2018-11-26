import numpy as np
import matplotlib.pyplot as plt 
# from quadprog import solve_qp
from sklearn import svm
from sklearn.model_selection import KFold

def runSVM(X, Y, Cs):
    #KFold cross validation
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)

    # Fit SVM model and predict for various C's
    accuraciesByC = []
    for c in Cs:
        model = svm.SVC(C=c, kernel='linear')
        accuracies = []
        for train_index, test_index in kf.split(X):
            model.fit(X[train_index], Y[train_index])
            prediction = model.predict(X[test_index])
            numCorrect = sum(map(lambda x, y: x==y, prediction.tolist(), Y[test_index].T.tolist()[0]))
            accuracy = numCorrect/len(prediction.tolist())
            accuracies.append(accuracy)
        accuraciesByC.append(np.mean(accuracies))
    return accuraciesByC


if __name__ == '__main__':
    # Generate linearly separable data
    X1 = np.random.multivariate_normal([25, 25], [[5, 0], [0, 5]], 100)
    X2 = np.random.multivariate_normal([35, 35], [[5, 0], [0, 5]], 100)
    X = np.concatenate((X1, X2), axis=0)

    Y1 = -1 * np.ones((1, 100)).T
    Y2 = np.ones((1, 100)).T
    Y = np.concatenate((Y1, Y2), axis=0) 

    plt.plot(X1.T[0], X1.T[1], '.')
    plt.plot(X2.T[0], X2.T[1], '.')
    plt.show()

    Cs = [i*0.1 for i in range(1, 20)]

    performance1 = runSVM(X, Y, Cs)
    print(performance1)


    # Generate noisy data
    X1 = np.random.multivariate_normal([25, 25], [[15, 0], [0, 15]], 100)
    X2 = np.random.multivariate_normal([35, 35], [[15, 0], [0, 15]], 100)
    X = np.concatenate((X1, X2), axis=0)

    Y1 = -1 * np.ones((1, 100)).T
    Y2 = np.ones((1, 100)).T
    Y = np.concatenate((Y1, Y2), axis=0) 

    plt.plot(X1.T[0], X1.T[1], '.')
    plt.plot(X2.T[0], X2.T[1], '.')
    plt.show()

    Cs = [i*0.1 for i in range(1, 100)]

    performance2 = runSVM(X, Y, Cs)
    print(performance2)

    plt.plot(Cs, performance2)
    plt.show()


    # Generate REALLY noisy data
    X1 = np.random.multivariate_normal([25, 25], [[35, 0], [0, 35]], 100)
    X2 = np.random.multivariate_normal([35, 35], [[35, 0], [0, 35]], 100)
    X = np.concatenate((X1, X2), axis=0)

    Y1 = -1 * np.ones((1, 100)).T
    Y2 = np.ones((1, 100)).T
    Y = np.concatenate((Y1, Y2), axis=0) 

    plt.plot(X1.T[0], X1.T[1], '.')
    plt.plot(X2.T[0], X2.T[1], '.')
    plt.show()

    Cs = [i*0.1 for i in range(1, 100)]

    performance3 = runSVM(X, Y, Cs)
    print(performance3)

    plt.plot(Cs, performance3)
    plt.show()


######################################################
# Manual attempt  before realizing quadprog is broken
######################################################

    # # Quadratic programming
    # H = np.eye(2)
    # f = np.zeros((2, 1))
    # A = -1 * np.diag([200]) * X * Y
    # b = np.ones((2, 1))

    # w = solve_qp(H, f, A, b)
    # print(w)



