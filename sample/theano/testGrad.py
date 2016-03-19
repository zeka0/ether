import sample.theano
from sample.theano import tensor as T
import numpy as np

x = sample.theano.shared(np.ones((1, 10)))
y = (x**2).sum()
gx = T.grad(y, x)
fun = sample.theano.function(inputs=[], outputs=gx)
print x
print fun()