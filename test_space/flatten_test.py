import theano
from theano import tensor as T
import numpy as np

X = T.tensor3('X3')
X1 = T.flatten(X, outdim=1)
X2 = T.flatten(X, outdim=2)
X3 = T.flatten(X, outdim=3)

f1 = theano.function(inputs=[X], outputs=X1)
f2 = theano.function(inputs=[X], outputs=X2)
f3 = theano.function(inputs=[X], outputs=X3)

x = np.arange(0,27).reshape((3,3,3))
print 'x\n', x
print 'ndim=1\n', f1(x)
print 'ndim=2\n', f2(x)
print 'ndim=3\n',f3(x)
