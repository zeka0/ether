import theano
from theano import tensor as T
import numpy as np
import pickle
from nnet.util.shape import *
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

x = T.matrix()
y = x.dimshuffle('x', 1, 0)
f = theano.function(inputs=[x], outputs=y)
z = np.arange(0,10).reshape((2, 5))
print f(z)
