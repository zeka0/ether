from core import *


class RBFLayer(layer):
    '''
    This class implements the naivest version of the RDF
    '''
    def __init__(self):
        layer.__init__(self)

    def init_kernels(self, numOfConnections):
        self.kernels = uniform.init_kernels(numOfConnections)

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

class fixedRBFLayer(RBFLayer):
    '''
    Uses user defined kernels and during the training, the kernels are not changed
    However, the computation may report an mis-match error if user defined kernels with wrong shape
    '''
    def __init__(self, kernels):
        RBFLayer.__init__(self)
        self.kernels = kernels

    def init_kernels(self, numOfConnections):
        '''
        Ignore the input
        '''
        pass

    def get_params(self):
        return None

class GassinRBFLayer(RBFLayer):
    def __init__(self):
        RBFLayer.__init__(self)

    def get_RBFTensor(self):
        return ( T.pow( self.get_kernels() - self.get_inputTensor(), 2 ) ).sum()

class fixedGassinRBFLayer(fixedRBFLayer):
    def __init__(self, kernels):
        fixedRBFLayer.__init__(self, kernels)

    def get_RBFTensor(self):
        return ( T.pow( self.get_kernels() - self.get_inputTensor(), 2 ) ).sum()