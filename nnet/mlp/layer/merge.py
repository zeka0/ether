from core import *
from nnet.util.shape import flatten_shape

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
        #TODO: support mini_batch
        return ( 1, len(self.get_preLayers()) * self.get_inputShape()[0] )

    def get_inputShape(self):
        return self.inputShape

    def get_params(self):
        return None

    def verify_shape(self):
        baseShape = self.get_preLayers()[0].get_outputShape()
        for layer in self.get_preLayers():
            if layer.get_outputShape() != baseShape:
                raise shapeError(self, 'shape mis-match')

    def connect(self, *layers):
        self.preLayers = layers
        inputTensors = []
        for layer in layers:
            inputTensors.append( layer.get_outputTensor() )
        if inputTensors[0].type == T.dscalar:
            outputTensor = T.stack(*inputTensors)
        else:
            outputTensor = T.concatenate( inputTensors, axis=1 )
        self.set_outputTensor( outputTensor )
        self.set_inputTensors( inputTensors )
        self.inputShape = layers[0].get_outputShape()

class flattenLayer(layer):
    def __init__(self):
        layer.__init__(self)

    def verify_shape(self):
        pass

    def connect(self, *layers):
        assert len(layers) == 1
        self.set_inputTensor( layers[0].get_outputTensor() )
        outputTensor = T.flatten( self.get_inputTensor(), outdim=1 )
        self.inputShape = layers[0].get_outputShape()
        self.outputShape = flatten_shape(self.inputShape)
        outputTensor = T.reshape(outputTensor, newshape=self.outputShape)
        self.set_outputTensor(outputTensor)

    def get_params(self):
        return None

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.outputShape
