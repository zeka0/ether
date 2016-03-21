from numpy.matlib import *
from theano import tensor as T
import theano
import numpy as np

def init_uniform(shape):
    if isinstance(shape, tuple):
        return theano.shared( rand(shape) )
    else:
        return theano.shared( rand(1, shape) )

def init_filters(numOfFilters, filterShape):
    '''
    Return a list of filters
    '''
    filterList = []
    for i in xrange(numOfFilters):
        filterList.append( theano.shared( rand(filterShape) ) )
    return filterList

def init_weights(weightsShape):
    assert len(weightsShape) == 2
    return theano.shared( rand(weightsShape) )

def init_kernels(numOfKernels):
    return init_uniform(numOfKernels)

def init_bias():
    return theano.shared( float(rand((1,))) )
