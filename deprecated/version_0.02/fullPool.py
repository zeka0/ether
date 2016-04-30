from collections import deque

from ether.util import *
class fullPool(object):
    '''
    Full sampling pool
    '''
    def __init__(self, instances, sampleRate=0.1, isTrain=True, maxLen=100):
        self.train_instances=deque(maxlen=maxLen)
        self.validity_instances=deque(maxlen=maxLen)
        fullPool.set_instances(self, instances, isTrain, sampleRate)

    def pop_deque(self, batchSize=1):
        for i in xrange(batchSize):
            if self.isTrain:
                yield self.train_instances.pop()
            else:
                yield self.validity_instances.pop()

    def set_state(self, isTrain):
        '''
        Settings for specify whether pool is in training state
        '''
        self.isTrain=isTrain

    def get_max_bufferSize(self):
        if self.isTrain:
            return self.train_instances.maxlen
        else:
            return self.validity_instances.maxlen


    def get_bufferSize(self):
        if self.isTrain:
            return len(self.train_instances)
        else:
            return len(self.validity_instances)

    def has_nextInstance(self, batchSize=1):
        if self.isTrain:
            return len(self.train_instances) != 0
        else:
            return len(self.validity_instances) != 0

    def get_nextInstance(self, batchSize=1):
        '''
        Always returns a list
        '''
        tmpHold=[]
        tmpHold.extend(self.pop_deque(batchSize))
        if len(tmpHold)>0:
            return tmpHold
        else:
            raise instanceException('no available instance')

    def set_instances(self, instances, isTrain, sampleRate=0.1):
        self.set_state(isTrain) #First set the status
        self.instances=instances #General pool of the instances
        self.sample_instances(sampleRate)

    def sample_instances(self, sampleRate=0.1):
        '''
        Sample from self.instances to form self.train_instances and self.validity_instances
        Use self.__sampleInstances(0) to disnable samling process
        '''
        if sampleRate<=0: #__sampleInstances is disnabled
            if self.isTrain:
                self.train_instances.extend(self.instances)
            else:
                self.validity_instances.extend(self.instances)
            return

        from numpy import random as rd
        nindices=rd.shuffle(xrange(len(self.instances)))

        #Seperate validity and train instances
        validity_indices=nindices[0:int(len(nindices)*sampleRate)] #Sample indices
        train_indices=nindices[len(validity_indices):]

        for index in validity_indices:
            self.validity_instances.append(self.instances[index])
        for index in train_indices:
            self.train_instances.append(self.instances[index])

    def is_stochastic(self):
        return False