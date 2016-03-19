import theano
from theano import tensor as T
import numpy as np

x = theano.shared( np.ones((1,10)) )
y = x + 1
fun = theano.function(inputs=[], outputs=[], updates=[(x, y)])
zz = []
for i in xrange(10):
    zz.append(x.get_value())
    fun()
from pprint import pprint
pprint(zz)