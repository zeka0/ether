import theano
from theano import tensor as T
import numpy as np
from numpy.matlib import *
from theano.tensor.signal.conv import *
import gzip
import pickle
from nnet.instance.pool.filter import picFilter
from numpy.matlib import rand
def readMnist(filePath):
    '''
    Caution!
    The data loaded using this module may not compatible with the mnistDataReader
    Because the data-set x may be pre-divided into train-test sets
    '''
    with gzip.open( filePath, 'rb' ) as f:
        x = pickle.load(f)
        return x

if __name__ == '__main__':
    xData = readMnist( r'E:\VirtualDesktop\mnist.pkl.gz' )
    xImags = np.vstack( ( xData[0][0], xData[1][0] ) )
    xTargs = np.hstack( ( xData[0][1], xData[1][1] ) )
    xUniqTargs = np.unique( xTargs )

    xfilter = theano.shared(rand((14, 14)))
    input = T.matrix( 'imag')
    conv = conv2d( input=input, filters=xfilter, subsample=(1, 1) )
    convFunc = theano.function(inputs=[input], outputs=conv)
    print 'convResulit#', convFunc( xImags[0] )
    print 'convResulit#', convFunc( xImags[1] )
    print 'convResulit#', convFunc( xImags[12] )
