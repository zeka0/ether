import sample.theano
from sample.theano import tensor as T
import numpy as np
from numpy.matlib import *
from nnet.util import *

w = rand( (10, 1) )
x = rand( (1, 10) )
xT = T.matrix()
wT = T.matrix()
zT = xT.dot(wT)
fun = test.theano.function(inputs=[xT, wT], outputs=zT)
print fun(x, w)
