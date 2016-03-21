import theano

from theano import tensor as T
import numpy as np
from numpy.matlib import *
from theano.tensor.signal.conv import *
from nnet.util.activation import *
from nnet.mlp.initialize import *

xT = T.matrix()
yT = sigmoid(xT)

fun = theano.function(inputs=[xT], outputs=yT)
for i in xrange(3):
    print fun(rand(1, 10))