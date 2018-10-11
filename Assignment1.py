import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats


######### Linearly Seperable #########

#Class 1 
mean1 = [0,0]
cov1 = [[1, 0], [0, 2]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 50).T

# Class 2 
mean2 = [5,5]
cov2 = [[1, 0], [0, 2]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 50).T

plt.figure()
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'o')
plt.axis('equal')
plt.title('Linearly Separable')
plt.show()


######### Non-linearly Seperable #########

numElements = 200
s = 5

#Class 1 
mean1 = [5, 5]
cov1 = [[10, 0], [0, 10]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, numElements).T

x1 = x1.tolist()
y1 = y1.tolist()
tx1 = []
ty1 = []

for n in range(1, numElements):
    if not (x1[n] < 5+s and x1[n] > 5-s and y1[n] < 5+s and y1[n] > 5-s):
        tx1.append(x1[n])
        ty1.append(y1[n])

x1 = tx1
y1 = ty1

# Class 2 
mean2 = [5,5]
cov2 = [[1, 0], [0, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, numElements).T

plt.figure()
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'o')
plt.axis('equal')
plt.title('Nonlinearly Separable')
plt.show()


######### Highly Correlated #########

# Class 1 
mean1 = [1, 1]
cov1 = [[0.5, 1], [10, 10]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 50).T

# Class 2 
mean2 = [8.25 ,6]
cov2 = [[0.5, 1], [10, 10]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 50).T

plt.figure()
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'o')
plt.axis('equal')
plt.title('Highly Correlated')
plt.show()


######### BiModal #########

# Class 1 
mean1_1 = [1, 1]
cov1_1 = [[1, 0], [0,1]]
mean1_2 = [10, 10]
cov1_2 = [[1, 0], [0, 1]]

D1 = np.random.multivariate_normal(mean1_1, cov1_1, 50)
D2 = np.random.multivariate_normal(mean1_2, cov1_2, 50)
x1, y1 = np.concatenate((D1, D2)).T

# Class 2 
mean2 = [5,5]
cov2 = [[1, 0], [0, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 50).T

plt.figure()
plt.plot(x1, y1, 'x')
plt.plot(x2, y2, 'o')
plt.axis('equal')
plt.title('Bimodal')
plt.show()
