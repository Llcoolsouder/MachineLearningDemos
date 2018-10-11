import numpy as np
import pandas
from sklearn.naive_bayes import GaussianNB

# 1=Setosa, 2=Versicolour, 3=Virginica
data = pandas.read_csv('../Data/Iris.csv')
numRows = 150

trainingData = data.iloc[0:105]
testData = data.iloc[105:numRows, :]

trainingLabels = trainingData.iloc[:, 4].values
trainingParameters = trainingData.iloc[:, 0:3].values
testLabels = testData.iloc[:, 4].values
testParameters = testData.iloc[:, 0:3].values

gnb = GaussianNB()
gnb.fit(trainingParameters, trainingLabels)
predictions = gnb.predict(testParameters)

numCorrect = (testLabels == predictions).sum()
numTested = testLabels.shape[0]

print(predictions)
print(testLabels)

print('Correctly classified',  numCorrect/numTested * 100, ' percent of irises')