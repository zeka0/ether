import theano
from theano import tensor as T
import numpy as np
import pickle
from nnet.util.shape import *

x = np.ones((3,4))
y = 3*x
print type(x)
print type(y)
print 'End'
