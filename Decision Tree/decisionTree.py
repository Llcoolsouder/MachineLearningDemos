from sklearn import tree
# import graphviz
import pandas
import numpy as np


# Load Data
df_train = pandas.read_csv('../Data/Dota2/dota2Train.csv')
X_train = df_train.iloc[:, 1:].values
Y_train = df_train.iloc[:, 0].values

df_test = pandas.read_csv('../Data/Dota2/dota2Test.csv')
X_test = df_test.iloc[:, 1:].values
Y_test = df_test.iloc[:, 0].values

# Instantiate classifier
model = tree.DecisionTreeClassifier()
model.fit(X_train, Y_train)

# Evaluate

### This part takes FOREVER
# tree_data = tree.export_graphviz(model, out_file=None)
# graph = graphviz.Source(tree_data)
# print('Rendering tree...')
# graph.render("Dota2")
# print('Done')

prediction = model.predict(X_test)
numCorrect = sum(map(lambda truth, pred: truth==pred, Y_test.T.tolist(), prediction.T.tolist()))
totalPreds = len(prediction.T.tolist())
print(numCorrect/totalPreds * 100)
