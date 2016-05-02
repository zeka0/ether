import theano
from theano import tensor as T
import numpy as np
import pickle
from ether.util.shape import *
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

x = np.ones((2,3))
z = np.array(x.flat[:].T, ndmin=2)
print z.shape
