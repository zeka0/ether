
import theano

from theano import tensor as T
import numpy as np
import pickle
from ether.util.shape import *
from theano.tensor.signal.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from ether.component.init import *

nh = 100
Wx = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (1, nh)).astype(theano.config.floatX))
Wh = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (nh, nh)).astype(theano.config.floatX))
Wy = theano.shared(0.2 * np.random.uniform(-1.0, 1.0, (nh, 1)).astype(theano.config.floatX))
bh = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
by = theano.shared(np.zeros(1, dtype=theano.config.floatX))
h0 = theano.shared(np.zeros(nh, dtype=theano.config.floatX))
x = T.matrix()

def recurrence(x_t, h_tm1):
    ha_t = T.dot(x_t, Wx) + T.dot(h_tm1, Wh) + bh
    h_t = T.tanh(ha_t)
    s_t = T.dot(h_t, Wy) + by
    return [ha_t, h_t, s_t]

([ha, h, activations], updates) = theano.scan(fn=recurrence, sequences=x, outputs_info=[dict(), h0, dict()])

print ha
print h
print activations
print updates

x = [T.matrix(), T.matrix(), T.matrix()]
y = 1
for i in xrange(len(x)):
    y = y * x[i]
y = y.sum()
yz = [y, y, y]
z = [T.vector(), T.vector(), T.vector()]
u = T.Rop(y, x, z)
print u

