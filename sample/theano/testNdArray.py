import theano
from theano import tensor as T
import numpy as np

x = np.ones((4, 10), dtype=np.uint8)
y = (x == 1)
x[y] = 22
y = (y == False)
print x
print y
