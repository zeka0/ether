import theano
import numpy as np

def init_filters(numOfFilters, filterShape, mean=0, std=1e-2):
    '''
    Return a list of filters
    '''
    filterList = []
    for i in xrange(numOfFilters):
        filterList.append( theano.shared( np.random.normal(loc=mean, scale=std, size=filterShape) ) )
    return filterList
