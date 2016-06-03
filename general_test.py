
import theano
from theano import tensor as T
import numpy as np
import pickle
from ether.util.shape import *
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

from ether.component.init import *
yt = T.matrix()
xt = theano_rng.binomial(size=(1,8), p=0.5, dtype=theano.config.floatX)
rfn = theano.function(inputs=[], outputs=xt)
x = theano.shared( rfn() )

zt = T.sum(x * yt)
ct = T.grad(zt, yt)
fn = theano.function(inputs=[yt], outputs=ct)
print x.get_value()
print fn(np.ones((1, 8)))

x.set_value(np.zeros((1,8)))
print x.get_value()
print fn(np.ones((1,8)))

class X:
    def __call__(self, *args, **kwargs):
        input = kwargs['input']
        print input**2

h = X()
h(input=10)
