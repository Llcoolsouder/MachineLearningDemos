from sklearn.ensemble import RandomForestClassifier
import pandas

# Load Data
df_train = pandas.read_csv('../Data/Dota2/dota2Train.csv')
X_train = df_train.iloc[:, 1:].values
Y_train = df_train.iloc[:, 0].values

df_test = pandas.read_csv('../Data/Dota2/dota2Test.csv')
X_test = df_test.iloc[:, 1:].values
Y_test = df_test.iloc[:, 0].values

# Instantiate Classifier
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# Evaluate
prediction = model.predict(X_test)
numCorrect = sum(map(lambda truth, pred: truth==pred, Y_test.T.tolist(), prediction.T.tolist()))
totalPreds = len(prediction.T.tolist())
print(numCorrect/totalPreds * 100)