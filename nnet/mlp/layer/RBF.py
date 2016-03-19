from core import *
from nnet.mlp.initialize import *
from nnet.util.shape import *

class RBFLayer(layer):
    '''
    This class implements the naivest version of the RDF
    '''
    def __init__(self):
        layer.__init__(self)

    def init_kernels(self, numOfConnections):
        self.kernels = uniform.init_kernels( numOfConnections )

    def get_kernels(self):
        return self.kernels

    def get_params(self):
        return [self.get_kernels()]

    def connect(self, *layers):
        assert len(layers) == 1
        self.intputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.init_kernels( self.get_inputShape()[1] ) #assume inputShape is 1D
        outputTensor = self.get_RBFTensor()
        self.set_outputTensor( outputTensor )

    def get_inputShape(self):
        return self.intputShape

    def get_outputShape(self):
        return (1, 1)

    def verify_shape(self):
        pass

    def get_RBFTensor(self):
        '''
        Subclass should modify this to return a tensor indicating the function it uses as the base function
        '''
        raise NotImplementedError()

class GassinRBFLayer(RBFLayer):
    def __init__(self):
        RBFLayer.__init__(self)

    def get_RBFTensor(self):
        return ( T.pow( self.get_kernels() - self.get_inputTensor(), 2 ) ).sum()
