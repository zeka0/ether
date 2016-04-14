import theano
import numpy as np

def transform_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, int):
        return (1, shape)

def init_shared(**kwargs):
    '''
    distr means distribution
    '''
    assert kwargs.has_key('distr')
    if kwargs['distr'] == 'normal':
        return normal_init_shared(**kwargs)
    elif kwargs['distr'] == 'uniform':
        return uniform_init_shared(**kwargs)
    elif kwargs['distr'] == 'scala':
        return scala_init_shared(**kwargs)
    elif kwargs['distr'] == 'constant':
        return constant_init_shared(**kwargs)
    else: raise NotImplementedError()

def normal_init_shared(**kwargs):
    shape = kwargs['shape']
    shape = transform_shape(shape)
    mean = kwargs['mean']
    std = kwargs['std']
    return theano.shared( np.random.normal(loc=mean, scale=std, size=shape) )

def uniform_init_shared(**kwargs):
    shape = kwargs['shape']
    shape = transform_shape(shape)
    low = kwargs['low']
    high = kwargs['high']
    return theano.shared( np.random.uniform(low=low, high=high, size=shape) )

def constant_init_shared(**kwargs):
    shape = kwargs['shape']
    shape = transform_shape(shape)
    value = kwargs['value']
    return theano.shared( value * np.ones(shape), size=shape )

def scala_init_shared(**kwargs):
    '''
    type should be of python type
    eg, int, double
    '''
    type = kwargs['type']
    value = kwargs['value']
    return theano.shared( type(value) )
