import pandas
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold


if __name__ == '__main__':
    # Import data
    data = pandas.read_csv('../Data/Iris.csv')

    X = data.iloc[:, 0:4].values
    Y = data.iloc[:, 4].values

    # K Fold Cross validation
    kf = KFold(n_splits=10)
    kf.get_n_splits(np.transpose(X))

    lReg = LogisticRegression(solver='lbfgs', max_iter=250000)

    accuracies = []
    for train_index, test_index in kf.split(X):
            # Train
            lReg.fit(X[train_index], Y[train_index])

            # Predict
            predict = lReg.predict(X[test_index])            

            # Compute accuracy for current "fold"
            numCorrect = sum(map(lambda x, y: x==y, predict, Y[test_index]))
            accuracy = numCorrect/len(predict) * 100
            accuracies.append(accuracy)

    print(accuracies)
    print(np.mean(accuracies))
    