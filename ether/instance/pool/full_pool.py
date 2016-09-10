from core import poolBase
import numpy as np

class fullPool(poolBase):
    '''
    If the sdRewind is True:
    This pool will rewind its self if the instances are not available.
    That means it will re-use the previous data.
    If it's not, then it will just behave like a deqPool.
    '''
    def __init__(self, data, sdRewind):
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
