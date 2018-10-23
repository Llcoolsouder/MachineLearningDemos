import numpy as np 
import matplotlib.pyplot as plt 

def h(theta, X):
    # [1x3] * [3xN] = [1xN]
    # return 1 / (1 + np.exp(-np.transpose(theta) * X))
    return 1/(1 + np.exp(-np.transpose(theta) * X))

def logistic_regression(X_train, Y_train, max_iterations=25000, a=0.0001, stopping_cond=0.01):
    i = 0                   # Iteration counter
    theta = np.ones((3,1))  # Logistic function parameters
    dJ = np.ones((3, 1))    # Derivative of cost function with respect to `theta`
    while (abs(np.max(dJ)) > stopping_cond and i<max_iterations):
        # [1xN] * [Nx3] = [1x3]
        dJ =  X_train * np.transpose(h(theta, X_train) - Y_train)
        theta -= a * dJ
        i += 1
    return theta


if __name__ == '__main__':
    # Generate data
    X1 = np.random.multivariate_normal([25, 25], [[5, 0], [0, 5]], 100).T
    X2 = np.random.multivariate_normal([35, 35], [[5, 0], [0, 5]], 100).T

    plt.plot(X1[0], X1[1], '.')
    plt.plot(X2[0], X2[1], '.')
    plt.show()

    # Data is in the format: [X1, X2, 1, Y]
    X = np.concatenate((X1, X2), axis=1)
    X = np.concatenate((X, np.ones((1, 200))), axis=0)
    Y = np.concatenate((np.zeros((1, 100)), np.ones((1, 100))), axis=1)
    data = np.concatenate((X, Y), axis=0)
    np.random.shuffle(np.transpose(data))

    data_train = data[:, :int(data.shape[1] * 0.7)]
    data_test = data[:, int(data.shape[1] * 0.7):]
    X_train = np.matrix(data_train[:3, :])
    Y_train = np.matrix(data_train[3, :])
    X_test = np.matrix(data_test[:3, :])
    Y_test = np.matrix(data_test[3, :]).tolist()[0]

    # Do logistic regression
    theta = logistic_regression(X_train, Y_train, 25000, 0.0001, 0.01)

    # Calculate accuracy and plot predictions
    predictionP = h(theta, X_test).tolist()[0]
    print(predictionP)
    prediction = list(map(lambda x: 1 if x >= 0.5 else 0, predictionP))
    percentCorrect = sum(map(lambda p, y: p==y, prediction, Y_test))/len(prediction) * 100
    print(percentCorrect)

    pred0 = []
    pred1 = []
    for i in range(X_test.shape[1]):
        if Y_test[i] == 0:
            pred0.append(tuple(X_test[:2, i].tolist()))
        elif Y_test[i] == 1:
            pred1.append(tuple(X_test[:2, i].tolist()))

    pred0x, pred0y = zip(*pred0)
    pred1x, pred1y = zip(*pred1)

    plt.plot(pred0x, pred0y, '.')
    plt.plot(pred1x, pred1y, '.')
    plt.show()