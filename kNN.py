import numpy as np
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

# 1=Setosa, 2=Versicolour, 3=Virginica
data = pandas.read_csv('Iris.csv')
numRows = data.values.shape[0]
labels = data.iloc[:, 4].values
parameters = data.iloc[:, 0:3].values

# K Fold Cross validation
kf = KFold(n_splits=10)
kf.get_n_splits(np.transpose(parameters))

performanceByK = []
for k in range(1, 21, 2):
    performances = []
    knn = KNeighborsClassifier(n_neighbors=k)
    for train_index, test_index in kf.split(parameters):
        # Predict
        knn.fit(parameters[train_index], labels[train_index])
        prediction = knn.predict(parameters[test_index])

        # Compute performance for current "fold"
        performances.append((prediction == labels[test_index]).sum()/labels[test_index].shape[0])
        print((prediction == labels[test_index]).sum()/labels[test_index].shape[0])
    # Average performance is overall performance for current K
    performanceByK.append(np.mean(performances))

plt.plot(range(1, 21, 2), performanceByK)
plt.xlabel('K')
plt.ylabel('Performance')
plt.xticks(range(1, 21, 2))
plt.title('Performance vs K')
plt.show()