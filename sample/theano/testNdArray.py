import sample.theano
from sample.theano import tensor as T
import numpy as np

x = np.ones((1, 10))
y = np.zeros(10)
print len(x)
print x.ndim
print len(y)
print y.ndim
x[0] = y
print x
print y
