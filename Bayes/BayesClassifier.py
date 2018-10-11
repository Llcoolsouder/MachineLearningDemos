import numpy as np
import matplotlib.pyplot as plt

# X is class {1, 2}
# Y is feature
# Finding P(X|Y) = P(Y|X) * P(X) / P(Y)

# Generate data
N = 2000
C1 = np.random.normal(25, 10, int(N/2)).T.tolist()
C2 = np.random.normal(75, 10, int(N/2)).T.tolist()
C = C1 + C2

C1_train = C1[0:int(len(C1)*0.7)]
C2_train = C2[0:int(len(C2)*0.7)]
C1_test = C1[int(len(C1)*0.7):len(C1)]
C2_test = C2[int(len(C2)*0.7):len(C2)]

# Ensure same bins are being used for all classes
cbin = plt.hist(C1_train + C2_train, 50)
bin_counts = cbin[0]
bin_edges = cbin[1]
plt.clf()

# Get Y for both classes
bin1 = plt.hist(C1_train, color='b', bins=bin_edges)
bin2 = plt.hist(C2_train, color='r', bins=bin_edges)
plt.show()

# Find P(Y) for all possible ym 
total_training_points = np.sum(bin_counts)
Py = list(map(lambda count: count/total_training_points, bin_counts))

# P(X) is 0.5 because I have an even number of samples from each class
Px = 0.5

# Find likelihood, P(Y|X)
Pyx = [[], []]

bin_counts1 = bin1[0]
total_C1_points = np.sum(bin_counts1)
Pyx[0] = list(map(lambda count: count/total_C1_points, bin_counts1))    # P(Y|X=1)

bin_counts2 = bin2[0]
total_C2_points = np.sum(bin_counts2)
Pyx[1] = list(map(lambda count: count/total_C2_points, bin_counts2))    # P(Y|X=1)

# Calculate posterior probability for all possible Y
Pxy = [[], []]
Pxy[0] = list(map(lambda Py, Pyx: Pyx*Px/Py if Py != 0 else 0, Py, Pyx[0]))
Pxy[1] = list(map(lambda Py, Pyx: Pyx*Px/Py if Py != 0 else 0, Py, Pyx[1]))

# Make all decisions now, so that we can simply check y's against a list later
D = list(map(lambda Pc1, Pc2: 1 if Pc1 > Pc2 else 2, Pxy[0], Pxy[1]))
print(D)

C1_predict = []
for testPoint in C1_test:
    for i in range(0, len(bin_edges)-1):
        if testPoint >= bin_edges[i] and testPoint < bin_edges[i+1]:
            C1_predict.append(D[i])

C2_predict = []
for testPoint in C2_test:
    for i in range(0, len(bin_edges)-1):
        if testPoint >= bin_edges[i] and testPoint < bin_edges[i+1]:
            C2_predict.append(D[i])   


# Calculate error
numberCorrect = sum(map(lambda x: x==1, C1_predict))
numberCorrect += sum(map(lambda x: x==2, C2_predict))
print('Classifier is ', numberCorrect/(len(C1_predict) + len(C2_predict))*100, 'percent correct')