import gzip
import pickle
from ether.instance.pool.filter import *
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
path = r'E:\VirtualDesktop\nnet\minist\flatten_double_mnist.pkl.gz'

if __name__ == '__main__':
    xData = readMnist( path )
    print 'End'
