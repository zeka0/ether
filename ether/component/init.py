import theano
import numpy as np
from theano import tensor as T

def transform_shape(shape):
    if isinstance(shape, tuple):
        return shape
    elif isinstance(shape, int):
        return (1, shape)

def init_input(inputShape):
    assert len(inputShape) >= 2 and len(inputShape) <=4
    if len(inputShape) == 2:
        inputTensor = T.matrix()
    elif len(inputShape) == 3:
        inputTensor = T.tensor3()
    else:
        inputTensor = T.tensor4()
    return inputTensor

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
    return theano.shared( value * np.ones(shape) )

def scala_init_shared(**kwargs):
    '''
    type should be of python type
    eg, int, double
    '''
    type = kwargs['type']
    value = kwargs['value']
    return theano.shared( type(value) )
