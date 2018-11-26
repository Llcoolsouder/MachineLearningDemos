import numpy as np 
import matplotlib.pyplot as plt 
import pandas
from sklearn import svm 
from sklearn.model_selection import KFold

def runSVM(X, Y, Cs, kern):
    #KFold cross validation
    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(X)

    # Fit SVM model and predict for various C's
    accuraciesByC = []
    for c in Cs:
        model = svm.SVC(C=c, kernel=kern)
        accuracies = []
        for train_index, test_index in kf.split(X):
            model.fit(X[train_index], Y[train_index])
            prediction = model.predict(X[test_index])                                              #Y[test_index].tolist()
            numCorrect = sum(map(lambda x, y: x==y, prediction.tolist(), Y[test_index].T.tolist()[0])) # Y[test_index].T.tolist()[0] for nonUCI. idk why right now
            accuracy = numCorrect/len(prediction.tolist())
            accuracies.append(accuracy)
        accuraciesByC.append(np.mean(accuracies))
    return accuraciesByC

if __name__ == '__main__':

    # Randomly generated non-linearly separable data
    X1_1 = np.random.multivariate_normal([10, 10], [[1, 1], [5, 10]], 100)
    X1_2 = np.random.multivariate_normal([15, 8], [[20, 5], [1, 1]], 100)
    X1 = np.concatenate((X1_1, X1_2), axis=0)
    X2 = np.random.multivariate_normal([15, 12], [[2, 0], [0, 2]], 200)
    X = np.concatenate((X1, X2), axis=0)

    Y1 = -1 * np.ones((200, 1))
    Y2 = np.ones((200, 1))
    Y = np.concatenate((Y1, Y2), axis=0)

    plt.plot(X1.T[0], X1.T[1], '.')
    plt.plot(X2.T[0], X2.T[1], '.')
    plt.show()

    Cs = [0.1*i for i in range(1, 100)]
    perf1 =  runSVM(X, Y, Cs, 'rbf')
    plt.plot(Cs, perf1)
    plt.title('Performance vs C for Gaussian Kernel')
    plt.show()

    perf2 =  runSVM(X, Y, Cs, 'poly')
    plt.plot(Cs, perf2)
    plt.title('Performance vs C for Polynomial(3) Kernel')
    plt.show()


    # UCI dataset
    #Load data
    data = pandas.read_csv('../Data/Iris.csv')
    X = data.iloc[:, 0:4].values
    Y = data.iloc[:, 4].values
    
    Cs = [0.1*i for i in range(1, 100)]
    uciPerf = runSVM(X, Y, Cs, 'rbf')
    plt.plot(Cs, uciPerf)
    plt.title('Performance vs C for Gaussian Kernel')
    plt.show()

    uciPerf = runSVM(X, Y, Cs, 'poly')
    plt.plot(Cs, uciPerf)
    plt.title('Performance vs C for Polynomial(3) Kernel')
    plt.show()