import theano
from theano import tensor as T
import numpy as np

x = T.matrix()
y = T.matrix()
z = x + y
fn = theano.function(inputs=[x, y], outputs=z)

data = np.arange(4).reshape((2,2))
print data
print '********'
print fn(data, data)
