import sample.theano
from sample.theano import tensor as T
import numpy as np
from numpy.matlib import *
from sample.theano.tensor.signal.conv import *

if __name__ == '__main__':
    x = T.matrix( 'x' )
    y = test.theano.shared(rand((10, 1)))
    z = y * x
    print 'x.shape#', x
    print 'z.shape#', z
