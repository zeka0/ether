import gzip
import pickle
from nnet.instance.pool.filter import *
def readMnist(filePath):
    '''
    Caution!
    The data loaded using this module may not compatible with the mnistDataReader
    Because the data-set x may be pre-divided into train-test sets
    '''
    with gzip.open( filePath, 'rb' ) as f:
        x = pickle.load(f)
        return x

import numpy as np

if __name__ == '__main__':
    xData = readMnist( r'E:\VirtualDesktop\lenet\mnist.pkl.gz' )
    xImags = np.vstack( ( xData[0][0], xData[1][0] ) )
    xTargs = np.hstack( ( xData[0][1], xData[1][1] ) )
    xUniqTargs = np.unique( xTargs )

    picf = picFilter(0, 255, True)
    f_xImage = picf.filter(xImags[0])
    print f_xImage
