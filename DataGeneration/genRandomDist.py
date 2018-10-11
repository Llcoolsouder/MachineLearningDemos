import numpy as np
import matplotlib.pyplot as plt

x = np.random.uniform(0, 1, 25)
print(x)

total = np.sum(x)
print(total)
x = x/total
print(x)
print(np.sum(x))

plt.stem(range(len(x)), x)
plt.show()
