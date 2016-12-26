import theano
from theano import tensor as T
import numpy as np

x = T.matrix()
y = T.sum(x, axis=1)
fn = theano.function(inputs=[x], outputs=T.mean(y, axis=0))
data = np.arange(4).reshape((2,2))
print data
print '******'
print data[0][0]
print data[0][1]
print fn(data)
