import theano
from theano import tensor as T
import numpy as np

W = [T.dmatrix('W1'), T.matrix('W2')]
V = [T.dmatrix('V1'), T.matrix('V2')]
x = T.dvector('x')
y = T.sum(T.dot(x, W[0]) + T.dot(x, W[1]))


Jv1 = T.Rop(y, W, V)
Jv2 = 0
y = y if y.ndim == 1 else T.flatten(y, outdim=1)
for Wi, Vi in zip(W, V):
    Jv2 += T.dot(T.flatten(theano.gradient.jacobian(y, Wi), outdim=2), T.flatten(Vi, outdim=1))
gj0 = theano.gradient.jacobian(y, W[0])
gj1 = theano.gradient.jacobian(y, W[1])

fun1 = theano.function(W + V + [x], Jv1)
fun2 = theano.function(W + V + [x], Jv2)

gfun0 = theano.function(W + [x], gj0)
gfun1 = theano.function(W + [x], gj1)

print 'orignal Rop'
print fun1([[1, 1], [1, 1]], [[1, 1], [1, 1]], [[2, 2], [2, 2]], [[2, 2], [2, 2]], [0,1])
print 'custom Rop'
print fun2([[1, 1], [1, 1]], [[1, 1], [1, 1]], [[2, 2], [2, 2]], [[2, 2], [2, 2]], [0,1])

print 'Jaco of W0'
print gfun0([[1, 1], [1, 1]], [[1, 1], [1, 1]], [0,1])
print 'Jaco of W1'
print gfun1([[1, 1], [1, 1]], [[1, 1], [1, 1]], [0,1])
