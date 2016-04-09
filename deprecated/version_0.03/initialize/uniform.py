from numpy.matlib import *
from theano import tensor as T
import theano
import numpy as np

def init_uniform(low, high, shape):
    if isinstance(shape, tuple):
        return theano.shared( rand(shape) * (high - low) + low )
    else:
        return theano.shared( rand(1, shape) * (high - low) + low )

def init_filters(numOfFilters, filterShape, low=0, high=1):
    '''
    Return a list of filters
    '''
    filterList = []
    for i in xrange(numOfFilters):
        filterList.append( init_uniform(low, high, filterShape) )
    return filterList

def init_kernels(numOfKernels, low=0, high=1):
    return init_uniform(low, high, numOfKernels)
