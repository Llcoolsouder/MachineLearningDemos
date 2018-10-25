import numpy as np 
import matplotlib.pyplot as plt 


if __name__ == '__main__':
    # Generate data
    X1 = np.random.multivariate_normal([25, 25], [[25, 0], [0, 25]], 100).T
    X2 = np.random.multivariate_normal([35, 35], [[25, 0], [0, 25]], 100).T

    plt.plot(X1[0], X1[1], '.', color='blue')
    plt.plot(X2[0], X2[1], '.', color='orange')
    plt.show()

    # Data is in the format: [X1, X2, 1, Y]
    X = np.concatenate((X1, X2), axis=1)
    X = np.concatenate((X, np.ones((1, 200))), axis=0)
    Y = np.concatenate((-1*np.ones((1, 100)), np.ones((1, 100))), axis=1)
    data = np.concatenate((X, Y), axis=0)
    np.random.shuffle(np.transpose(data))

    data_train = data[:, :int(data.shape[1] * 0.7)]
    data_test = data[:, int(data.shape[1] * 0.7):]
    X_train = np.matrix(data_train[:3, :])
    Y_train = np.matrix(data_train[3, :])
    X_test = np.matrix(data_test[:3, :])
    Y_test = np.matrix(data_test[3, :])

    # Do gradient descent for perceptron
    i = 0                           # Iteration counter
    w = np.ones((X.shape[0], 1))    # Perceptron weights
    dJ = np.ones((X.shape[0], 1))   # Derivative of cost function with respect to `theta`
    a = 0.0001
    while (abs(np.max(dJ)) > 0.01 and i<25000):
        predict = np.multiply(-w.transpose() * X_train, Y_train)
        isCorrect = list(map(lambda x: False if x<0 else True, predict.tolist()[0]))
        X_mismatch = [x for x, c in zip(X_train.transpose().tolist(), isCorrect) if not c]
        Y_mismatch = [y for y, c in zip(Y_train.transpose().tolist(), isCorrect) if not c]
        
        try:    # Found some mislabels
            dJ = np.sum(np.multiply(X_mismatch, Y_mismatch).transpose(), axis=1)
            dJ = dJ[:, np.newaxis]
            w -= a * dJ
            i += 1  
        
        except: # Found 0 mislabels
            break

    if (i==25000):
        print('Did not converge')

    # Calculate accuracy and plot predictions
    prediction = -w.transpose() * X_test
    print(prediction)
    prediction = list(map(lambda x: 1 if x > 0 else -1, prediction.tolist()[0]))
    percentCorrect = sum(map(lambda p, y: p==y, prediction, Y_test.tolist()[0]))/len(prediction) * 100
    print(percentCorrect)

    pred0 = []
    pred1 = []
    for i in range(X_test.shape[1]):
        if prediction[i] == -1:
            pred0.append(tuple(X_test[:2, i].tolist()))
        elif prediction[i] == 1:
            pred1.append(tuple(X_test[:2, i].tolist()))

    pred0x, pred0y = zip(*pred0)
    pred1x, pred1y = zip(*pred1)

    plt.plot(pred0x, pred0y, '.', color='blue')
    plt.plot(pred1x, pred1y, '.', color='orange')
    plt.show()

    # Illustrate decision boundary
    paintPoints = []
    for i in range(100):
        for j in range(100):
            paintPoints.append([i, j, 1])
    paintLabels = -w.transpose() * np.matrix(paintPoints).T
    paintLabels = list(map(lambda x: -1 if x<0 else 1, paintLabels.tolist()[0]))
    paint = list(map(lambda x, y: (x[0], x[1], x[2], y), paintPoints, paintLabels))
    paint0x, paint0y, _, _ = list(zip(*filter(lambda x: x[3]==-1, paint)))
    paint1x, paint1y, _, _ = list(zip(*filter(lambda x: x[3]==1, paint)))

    plt.plot(paint0x, paint0y, '.', color='blue')
    plt.plot(paint1x, paint1y, '.', color='orange')
    plt.title('Decision Boundary')
    plt.show()