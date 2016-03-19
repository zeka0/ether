__author__ = 'mac'

from collections import deque

from theanoNnet.nnetUtil.exception import *

class full_pool(object):
    '''
    Full sampling pool
    '''
    def __init__(self, instances, sampleRate=0.1, isTrain=True, maxLen=100):
        self.train_instances=deque(maxlen=maxLen)
        self.validity_instances=deque(maxlen=maxLen)
        full_pool.set_instances(self, instances, isTrain, sampleRate)

    def pop_deque(self, batchSize=1):
        for i in xrange(batchSize):
            if self.isTrain:
                yield self.train_instances.pop()
            else:
                yield self.validity_instances.pop()

    def set_state(self, isTrain):
        self.isTrain=isTrain #Settings for specify whether pool is in training state

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

class stochastic_pool(full_pool):
    '''
    Stochasticly drawing instances
    When using stochastic pool for testing(validitying)
    You should treat the output of this pool simply as testing instances
    '''
    def __init__(self, instances):
        full_pool.__init__(self, instances, 0, True) #disnable sampling

    def set_instances(self, instances):
        self.train_instances.extend(instances)

    def is_stochastic(self):
        return True

    def set_state(self, isTrain):
        self.isTrain = True

    def get_totalSize(self):
        raise NotImplementedError()
