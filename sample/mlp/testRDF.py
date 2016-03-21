import theano
from theano import tensor as T
from numpy.matlib import *
import numpy as np

x = np.ones( (1, 10) )
w = np.ones( (12, 10) )
xx = x
for i in xrange(11):
    xx = np.vstack((xx, x))
xT = T.matrix()
wT = T.matrix()
z = (wT - xT)
fun = theano.function(inputs=[xT, wT], outputs=z)
print fun(xx, w)