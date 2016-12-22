import numpy as np

x = np.ones(20).reshape((4,5))
print x
x = x.reshape((1, x.shape[0], -1))
print x
