import theano
from theano import tensor as T
import numpy as np

x = theano.shared(np.ones((1, 10)))
y = (x**2).sum()
gx = T.grad(y, x)
fun = theano.function(inputs=[], outputs=gx)
print x
print fun()