import numpy as np
import matplotlib.pyplot as plt

a0 = 0.0005
a1 = 0.0005

x = range(0, 100, 1)

m = 2
b = -1

y = list(map(lambda x: m*x + b + (np.random.rand()*5) - 2.5, x))

plt.plot(x, y, '.')
plt.title('Noisy Signal')
plt.show()

theta0 = 0 # m parameter
theta1 = 0 # b parameter

N = len(x)

dJ0 = 1
dJ1 = 1
# prevPos0 = True
# prevPos1 = True

thetas0 = []
thetas1 = []

while (abs(dJ0) > 0.01 or abs(dJ1) > 0.01):
    dJ0 = (1/N) * sum(list(map(lambda x, y: (y - (theta0*x + theta1))*(x), x, y)))
    # print (dJ0)

    dJ1 = (1/N) * sum(list(map(lambda x, y: (y - (theta0*x + theta1)), x, y)))
    # print(dJ1)

    # if (dJ0 > 0):   # +dJ0
    #     if (not prevPos0):
    #         a0 *= 0.75
    #     prevPos0 = True
    # else:           # -dJ0
    #     if (prevPos0):
    #         a0 *= 0.75
    #     prevPos0 = False

    # if (dJ1 > 0):   # +dJ1
    #     if (not prevPos1):
    #         a1 *= 0.75
    #     prevPos1 = True
    # else:           # -dJ1
    #     if (prevPos1):
    #         a1 *= 0.75
    #     prevPos1 = False

    theta0 += dJ0 * a0
    theta1 += dJ1 * a1
    thetas0.append(theta0)
    thetas1.append(theta1)
    # print(theta0)
    # print(theta1)
    # print(a0)
    # print(a1)
    # print('#################')

print(theta0)
print(theta1)

plt.plot(thetas0)
plt.show()

plt.plot(thetas1)
plt.show()

line = list(map(lambda x: theta0*x + theta1, x))
plt.plot(y, '.', color='blue')
plt.plot(line, color='red')
plt.show()
