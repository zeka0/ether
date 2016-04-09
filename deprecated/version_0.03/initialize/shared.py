import theano
import numpy as np
def shared_zeros(shape):
    return theano.shared( np.zeros(shape) )

def shared_ones(shape):
    return theano.shared( np.ones(shape) )

def shared_scalar(value):
    if isinstance(value, np.ndarray):
        return theano.shared( int(value) )
    else:
        return theano.shared(value)

def shared_double(value):
    if isinstance(value, np.ndarray):
        return theano.shared( float(value) )
    elif isinstance(value, int):
        return theano.shared(float(value))
    else:
        return theano.shared(value)
