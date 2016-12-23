import theano
from theano import tensor as T
import numpy as np

x = T.matrix()
y = T.matrix()

def elmentalwise_add(xi, yi):
    return xi + yi

res, updates = theano.scan(elmentalwise_add, sequences=[x, y])
fn = theano.function(inputs=[x, y], outputs=res)
print fn(np.arange(3).reshape((1,3)), np.arange(3).reshape((1,3)))
