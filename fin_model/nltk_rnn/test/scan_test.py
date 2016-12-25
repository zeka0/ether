import numpy as np
import theano
from theano import tensor as T

def print_x(x):
    print x
    return x

seq = T.matrix('x')

[xt], updates = theano.scan(print_x, sequences=[seq], outputs_info=None)
fn = theano.function(inputs=[seq], outputs=[xt], updates=updates)
fn(np.ones((4,4)))

