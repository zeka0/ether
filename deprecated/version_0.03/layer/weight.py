from core import *
from ether.util.shape import *


class weightLayer(layer):
    '''
    We can view weight-layer as a conv1D layer
    '''
    def __init__(self, numOfOutput):
        layer.__init__(self)
        self.numOfOutput = numOfOutput
        self.init_bias()

    def init_weights(self):
        #TODO:
        self.weights = uniform.init_uniform(low=-0.25, high=0.25, shape=self.get_weightMatrixShape())

    def init_bias(self):
        self.bias = shared.shared_double(0)

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.init_weights()
        outputTensor = self.get_inputTensor().dot( self.get_weights() ) + self.get_bias()
        self.set_outputTensor(outputTensor)

    def get_inputShape(self):
        return self.inputShape

    def get_weightMatrixShape(self):
        return weightMatrix_shape( self.get_inputShape(), self.numOfOutput)

    def get_outputShape(self):
        return ( 1, self.numOfOutput )

    def get_params(self):
        paramList = []
        paramList.append( self.get_weights() )
        paramList.append( self.get_bias() )
        return paramList

    def verify_shape(self):
        if len( self.get_inputShape() ) == 2:
            assert self.get_inputShape()[0] == 1
