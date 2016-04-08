import theano

from theano import tensor as T
import numpy as np
from numpy.matlib import *
from theano.tensor.signal.conv import *
from nnet.util.activation import *
from nnet.mlp.initialize import *

xT = T.matrix()
yT = softmax(xT)

fun = theano.function(inputs=[xT], outputs=yT)
print fun(np.array([[ 7037.5136159, 7611.69078938, 7197.10745514, 7373.56197423, 7395.85992396, 7241.17424998, 7202.14100706, 7449.58107226, 7190.49475249, 7514.04714423]]))
