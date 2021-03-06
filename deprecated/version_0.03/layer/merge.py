from core import *

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

    def get_outputShape(self):
        return ( 1, len(self.get_preLayers()) * self.get_inputShape()[0] )

    def get_inputShape(self):
        return self.inputShape

    def get_params(self):
        return None

    def verify_shape(self):
        baseShape = self.get_preLayers()[0].get_outputShape()
        if len(baseShape) == 2:
            assert baseShape[0] == 1 #we treat vector as a horizontal one
        for layer in self.get_preLayers():
            if layer.get_outputShape() != baseShape:
                raise shapeError(self, 'shape mis-match')
