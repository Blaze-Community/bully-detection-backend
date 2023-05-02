import numpy as np

y = [[5.1, 5.1, 5.1]]

maxElement = np.amax(y)
print(maxElement)
result = np.where(y == maxElement)
print(result, result[0], result[1])
print(result[1][0])