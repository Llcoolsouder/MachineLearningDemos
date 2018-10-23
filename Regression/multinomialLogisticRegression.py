# get MY functions
from logisticRegression import h, logistic_regression

# get third party libs
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

if __name__ == '__main__':
    # Generate data
    N = 100
    C0 = np.random.multivariate_normal([25, 25], [[5, 0], [0, 5]], N).T
    C1 = np.random.multivariate_normal([35, 35], [[5, 0], [0, 5]], N).T
    C2 = np.random.multivariate_normal([25, 35], [[5, 0], [0, 5]], N).T

    plt.plot(C0[0], C0[1], '.')
    plt.plot(C1[0], C1[1], '.')
    plt.plot(C2[0], C2[1], '.')
    plt.show()

    # Data is in the format: [X1, X2, 1, Y]
    ## Data used to determine C0 or !C0 (!C0 includes both C1 and C2)
    C = np.concatenate((C0, C1, C2), axis=1)
    C = np.concatenate((C, np.ones((1, N*3))), axis=0)
    Y_0vAll = np.concatenate((np.zeros((1, N)), np.ones((1, 2*N))), axis=1)
    Y_full  = np.concatenate((np.zeros((1, N)), np.ones((1, N)), 2 * np.ones((1, N))), axis=1)
    data_0vAll = np.concatenate((C, Y_0vAll), axis=0)
    data_0vAll = np.concatenate((data_0vAll, Y_full), axis=0)
    np.random.shuffle(np.transpose(data_0vAll))

    data_0vAll_train = data_0vAll[:, :int(data_0vAll.shape[1]*0.7)]
    data_0vAll_test = data_0vAll[:, int(data_0vAll.shape[1]*0.7):]
    X_0vAll_train = np.matrix(data_0vAll_train[:3, :])
    Y_0vAll_train = np.matrix(data_0vAll_train[3, :])
    X_test = np.matrix(data_0vAll_test[:3, :])
    Y_test = np.matrix(data_0vAll_test[4, :])

    ## Data used to determine C1 or C2
    C = np.concatenate((C1, C2), axis=1)
    C = np.concatenate((C, np.ones((1, N*2))), axis=0)
    Y_1v2 = np.concatenate((np.zeros((1, N)), np.ones((1, N))), axis=1)
    data_1v2 = np.concatenate((C, Y_1v2), axis=0)
    np.random.shuffle(np.transpose(data_1v2))
    
    data_1v2_train = data_1v2[:, :int(data_1v2.shape[1]*0.7)]
    data_1v2_test = data_1v2[:, int(data_1v2.shape[1]*0.7):]
    X_1v2_train = np.matrix(data_1v2_train[:3, :])
    Y_1v2_train = np.matrix(data_1v2_train[3, :])
    X_1v2_test = np.matrix(data_1v2_test[:3, :])
    Y_1v2_test = np.matrix(data_1v2_test[3, :])

    # Train
    theta_0 = logistic_regression(X_0vAll_train, Y_0vAll_train)
    theta_1 = logistic_regression(X_1v2_train, Y_1v2_train)

    # Predict
    predict_0 = h(theta_0, X_test).tolist()[0]
    predict_1 = h(theta_1, X_test).tolist()[0]
    predict_0 = list(map(lambda x: 0 if x<0.5 else 1, predict_0))
    predict_1 = list(map(lambda x: 0 if x<0.5 else 1, predict_1))
    predict = list(map(lambda p0, p1: 0 if p0==0 else p0+p1, predict_0, predict_1))
    
    # Calculate accuracy
    numCorrect = sum(map(lambda x, y: x==y, predict, Y_test.tolist()[0]))
    accuracy = numCorrect / len(predict) * 100
    print(accuracy)

    # Illustrate decision boundary
    paintPoints = []
    for i in range(100):
        for j in range(100):
            paintPoints.append([i, j, 1])
    paintLabels0 = h(theta_0, np.matrix(paintPoints).transpose()).tolist()[0]
    paintLabels1 = h(theta_1, np.matrix(paintPoints).transpose()).tolist()[0]
    paintLabels0 = list(map(lambda x: 0 if x<0.5 else 1, paintLabels0))
    paintLabels1 = list(map(lambda x: 0 if x<0.5 else 1, paintLabels1))
    paintLabels = list(map(lambda p0, p1: 0 if p0==0 else p0+p1, paintLabels0, paintLabels1))
   
    paint = list(map(lambda x, y: (x[0], x[1], x[2], y), paintPoints, paintLabels))
    paint0x, paint0y, _, _ = list(zip(*filter(lambda x: x[3]==0, paint)))
    paint1x, paint1y, _, _ = list(zip(*filter(lambda x: x[3]==1, paint)))
    paint2x, paint2y, _, _ = list(zip(*filter(lambda x: x[3]==2, paint)))

    plt.plot(paint0x, paint0y, '.')
    plt.plot(paint1x, paint1y, '.')
    plt.plot(paint2x, paint2y, '.')
    plt.title('Decision Boundary')
    plt.show()