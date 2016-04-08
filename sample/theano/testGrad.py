import theano
from theano import tensor as T
import numpy as np

x = theano.shared(np.ones((1, 10)))

y = (x**2).sum()
gx = T.grad(y, x)

xupdate = x - gx
fun = theano.function(inputs=[], outputs=[], updates=[(x, xupdate)])
print x.get_value()
print fun()
print x.get_value()