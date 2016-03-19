import theano
from theano import tensor as T
import numpy as np

def initWeightMatrix(rowSize, colSize):
    return np.ones((rowSize, colSize))

def initBias(shape=1): #Broadcastable
    return np.ones(shape)

class nnetController:
    def __init__(self):
        pass

    def set_owner(self, nnet):
        '''
        Set the owner of this optimizer
        Called by nnet automatically
        '''
        self.nnet=nnet

    def has_nextInstance(self, batchSize=1):
        return self.nnet.has_nextInstance(batchSize)

    def get_inputTensor(self):
        return self.nnet.get_inputTensor()

    def get_outputTensor(self):
        return self.nnet.get_outputTensor()

    def get_layerOutputTensors(self):
        return self.nnet.get_layerOutputTensors()

    def get_params(self):
        return self.nnet.get_params()

    def get_nextInstance(self, batchSize=1):
        return self.nnet.get_nextInstance(batchSize)

    def get_targetTensor(self):
        return self.nnet.get_targetTensor()

    def set_state(self, isTrain):
        self.nnet.set_state(isTrain)

    def predict(self, attrVec):
        return self.nnet.predict(attrVec)
