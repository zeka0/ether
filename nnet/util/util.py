import theano
from theano import tensor as T
import numpy as np

class nnetController:
    def __init__(self):
        pass

    def set_owner(self, nnet):
        '''
        Set the owner of this optimizer
        Called by nnet automatically
        '''
        self.nnet = nnet

    def get_inputTensor(self):
        return self.nnet.get_inputTensor()

    def get_outputTensor(self):
        return self.nnet.get_outputTensor()

    def get_layerOutputTensors(self):
        return self.nnet.get_layerOutputTensors()

    def get_params(self):
        return self.nnet.get_params()

    def get_targetTensor(self):
        return self.nnet.get_targetTensor()

    def predict(self, attrVec):
        return self.nnet.predict(attrVec)

    def get_layers(self):
        return self.nnet.get_layers()