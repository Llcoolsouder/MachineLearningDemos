from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas


if __name__ == '__main__':
    # Load data
    data = pandas.read_csv('../Data/Iris.csv')
    X = data.iloc[:, 0:4].values
    Y = data.iloc[:, 4].values
    Y = OneHotEncoder(sparse=False, categories='auto').fit_transform(Y.reshape(-1, 1))

    # Make NN model
    model = Sequential()
    model.add(Dense(4, input_dim=4, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(3, activation='sigmoid'))

    # Compile model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Evaluation
    kf = KFold(n_splits=10)
    kf.get_n_splits(np.transpose(X))

    accuracies = []
    for train_index, test_index in kf.split(X):
        # Train
        model.fit(X[train_index], Y[train_index], epochs=250)

        # Predict
        # predictions = model.predict(X[test_index])
        # print(predictions)
        evaluation = model.evaluate(X[test_index], Y[test_index])
        accuracies.append(evaluation[1])
        print(evaluation)
        
    print(accuracies)