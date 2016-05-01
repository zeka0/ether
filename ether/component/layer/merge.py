from core import *
from ether.util.shape import flatten_shape
import numpy as np

class merge1DLayer(layer):
    def __init__(self):
        layer.__init__(self)

    def get_preLayers(self):
        return self.preLayers

    def set_inputTensor(self, inputTensor):
        raise NotImplementedError('merge layer has multiple input-tensors')

    def get_inputTensor(self):
        raise NotImplementedError('merge layer has multiple input-tensors')

    def set_inputTensors(self, inputTensors):
        self.inputTensors = inputTensors

    def get_inputTensors(self):
        return self.inputTensors

    def get_outputShape(self):
        return ( self.inputShape[0], np.prod( self.inputShape[1:] ) )

    def get_inputShape(self):
        return self.inputShape

    def get_params(self):
        return []

    def connect(self, *layers):
        '''
        The outputTensor can't be a scala
        '''
        self.inputShape = layers[0].get_outputShape()
        for l in layers:
            assert l.get_ouputShape() == self.inputShape
        self.preLayers = layers
        inputTensors = []
        for layer in layers:
            inputTensors.append( layer.get_outputTensor() )
        outputTensor = T.concatenate( inputTensors, axis=1 )
        self.set_outputTensor( outputTensor )
        self.set_inputTensors( inputTensors )

class flattenLayer(layer):
    def __init__(self, outdim=2):
        '''
        The dim-1 means the number of mini-batch, and it can't be flattened
        '''
        layer.__init__(self)
        assert outdim >= 2
        self.outdim = outdim

    def connect(self, *layers):
        assert len(layers) == 1
        self.set_inputTensor( layers[0].get_outputTensor() )
        outputTensor = T.flatten( self.get_inputTensor(), outdim=self.outdim )
        self.inputShape = layers[0].get_outputShape()
        self.outputShape = flatten_shape(self.inputShape, outdim=self.outdim)
        self.set_outputTensor(outputTensor)

    def get_params(self):
        return []

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.outputShape
