import theano
from theano import tensor as T
import numpy as np
import pickle
from nnet.util.shape import *
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

xT = T.matrix()
yT = T.vector()
zT = xT - yT
f = theano.function(inputs=[xT, yT], outputs=zT)

x = np.ones((4,4))
y = np.ones((4,))
print x
print y
print f(x, y)
