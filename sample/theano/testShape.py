import theano
from theano import tensor as T
import numpy as np
from numpy.matlib import *
from theano.tensor.signal.conv import *

def testShape():
    x = T.matrix( 'x' )
    y = theano.shared(rand((10, 1)))
    z = y * x
    print 'x.shape#', x
    print 'z.shape#', z

if __name__ == '__main__':
    x = theano.shared(np.zeros((2,10)))
    y = theano.shared( float(rand((1,))) )
    z = y + x
    fun = theano.function(inputs=[], outputs=z)
    print fun()