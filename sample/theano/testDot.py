import theano
from theano import tensor as T
import numpy as np
from numpy.matlib import *
from nnet.util import *

w = zeros( (10, 10) )
x = zeros( (10, 10) )

dT = T.dscalar()
xT = T.matrix()
wT = T.matrix()

zT = xT.dot(wT) + dT
fun = theano.function(inputs=[xT, wT, dT], outputs=zT)
print fun(x, w, 1)
