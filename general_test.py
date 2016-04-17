import theano
from theano import tensor as T
import numpy as np
import pickle
from nnet.util.shape import *
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

x = np.ones((3,4))
print x.reshape((2,6))
