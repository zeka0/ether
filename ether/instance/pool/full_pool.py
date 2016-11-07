from core import poolBase
import numpy as np
from ether.instance.instance import instance

class fullInstancePool(poolBase):
    '''
    If the sdRewind is True:
    This pool will rewind its self if the instances are not available.
    That means it will re-use the previous data.
    If it's not, then it will just behave like a deqPool.
    '''
    def __init__(self, data, sdRewind):
        '''data should be of type instance'''
        self.data = data
        assert len(data) != 0
        self.sdRewind = sdRewind
        self.currPointer = 0

    def has_nextInstance_now(self, batchSize=1):
        if self.sdRewind:
            return True
        else:
            return len(self.data) - self.currPointer > batchSize

    def has_nextInstance(self, batchSize):
        return self.has_nextInstance_now(batchSize)

    def read_instances(self, batchSize=1):
        tmp = []
        if self.currPointer + batchSize - 1 < len(self.data):
            for i in xrange(batchSize):
                tmp.append( self.data[self.currPointer] )
                self.currPointer += 1
        else:
            sizeNow = batchSize
            while sizeNow > 0:
                if self.currPointer == len(self.data):
                    self.currPointer = 0
                tmp.append(self.data[self.currPointer])
                self.currPointer += 1
                sizeNow -= 1
        return tmp

class fullDataPool(poolBase):
    '''
    If the sdRewind is True:
    This pool will rewind its self if the instances are not available.
    That means it will re-use the previous data.
    If it's not, then it will just behave like a deqPool.
    '''
    def __init__(self, train, target, sdRewind):
        '''
        This version of full pool stores raw data in the pool, not instances
        '''
        assert len(train) == len(target)
        assert len(train) != 0
        self.train = train
        self.target = target
        self.sdRewind = sdRewind
        self.currPointer = 0

    def has_nextInstance_now(self, batchSize=1):
        if self.sdRewind:
            return True
        else:
            return len(self.train) - self.currPointer > batchSize

    def has_nextInstance(self, batchSize):
        return self.has_nextInstance_now(batchSize)

    def read_instances(self, batchSize=1):
        tmp = []
        if self.currPointer + batchSize - 1 < len(self.train):
            for i in xrange(batchSize):
                tmp.append( instance(self.train[self.currPointer], self.target[self.currPointer]) )
                self.currPointer += 1
        else:
            sizeNow = batchSize
            while sizeNow > 0:
                if self.currPointer == len(self.train):
                    self.currPointer = 0
                tmp.append( instance(self.train[self.currPointer], self.target[self.currPointer]) )
                self.currPointer += 1
                sizeNow -= 1
        return tmp
