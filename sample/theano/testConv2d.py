import sample.theano
from sample.theano import tensor as T
import numpy as np
from numpy.matlib import *
from sample.theano.tensor.signal.conv import *

if __name__ == '__main__':
    xfilter = test.theano.shared(rand((2, 3)))
    print 'xfilter#', xfilter.get_value()
    w = rand( ( 4, 6 ) )
    print 'w#', w
    input = T.matrix( 'imag')
    conv = conv2d( input=input, filters=xfilter, filter_shape=(2, 3), image_shape=(4, 6), subsample=(1, 1) )
    convFunc = test.theano.function(inputs=[input], outputs=[conv])
    print 'convResulit#', convFunc( w )