import theano
import numpy as np
def shared_zeros(shape):
    return theano.shared( np.zeros(shape) )

def shared_ones(shape):
    return theano.shared( np.ones(shape) )
