'''
The structure of the mnist data-set should be of a 2-dimensional ndarray
Of the shape (train-img, train-imd-target)
'''
from ether.instance.instance import *
from core import *

import gzip
import pickle
import numpy as np

def readMnist(filePath):
    '''
    Caution!
    The data loaded using this module may not compatible with the mnistDataReader
    Because the data-set x may be pre-divided into train-test sets
    '''
    with gzip.open( filePath, 'rb' ) as f:
        x = pickle.load(f)
        return x

class mnistDataReader(dataReader):
    def __init__(self, filePath, numOfTargets):
        xData = readMnist( filePath )
        xImags = np.vstack( ( xData[0][0], xData[1][0] ) )
        xTargs = np.hstack( ( xData[0][1], xData[1][1] ) )
        self.xImags = xImags
        self.xTargs = xTargs
        self.numOfTargets = numOfTargets
        self.currIndex = 0

    def get_numOf_Attrs(self):
        raise NotImplementedError('mnist doesn\'t have attributes' )

    def get_numOf_Targets(self):
        return self.numOfTargets

    def read_instance(self, batchSize=1):
        for i in xrange(batchSize):
            yield imgInstance( self.xImags[ self.currIndex ], self.xTargs[ self.currIndex ], numOfTars=self.numOfTargets )
            self.currIndex = self.currIndex + 1

    def has_nextInstance(self, size):
        return self.currIndex + size <= len( self.xTargs )

    def read_all(self):
        tmp = [ins for ins in self.read_instance(len(self.xTargs))]
        return tmp
