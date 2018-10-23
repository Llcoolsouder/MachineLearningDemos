import pandas
import numpy as np
from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import KFold

# I am assuming a certain file structure. If you move this script, change this line.
dataDir = '../Data/Gisette/'

# Import relevant data
trainingData = pandas.read_csv(dataDir + 'gisette_train.data', delim_whitespace=True).values
trainingLabels = pandas.read_csv(dataDir + 'gisette_train.labels', delim_whitespace=True).values
testData = pandas.read_csv(dataDir + 'gisette_valid.data', delim_whitespace=True).values
testLabels = pandas.read_csv(dataDir + 'gisette_valid.labels', delim_whitespace=True).values

# Set up logistic regression
lReg = LogisticRegression(solver='lbfgs', max_iter=250000)
lReg.fit(trainingData, trainingLabels.ravel())

# Predict
predictions = lReg.predict(testData)

# Analyze results
percentCorrect = sum(map(lambda x, y: x==y, predictions, testLabels))/len(predictions) * 100
print(percentCorrect)