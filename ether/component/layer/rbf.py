from core import *
from ether.component.init import init_shared

def gausi_rbf(rbflayer):
    return ( T.pow( rbflayer.get_kernel() - rbflayer.get_inputTensor(), 2 ) ).sum()

class RBFLayer(layer):
    '''
    This class implements the naivest version of the RDF
    '''
    def __init__(self, func, **kwargs):
        layer.__init__(self)
        self.rbfKwargs = kwargs
        self.func = func
        if len(kwargs) != 0:
            assert kwargs.has_key('kernel')
            self.kernelKwargs = kwargs['kernel']
            assert not self.kernelKwargs.has_key('shape') #No need to provide shape

    def init_kernel(self, numOfConnections):
        self.kernel = init_shared(shape=numOfConnections, **self.kernelKwargs)

    def get_kernel(self):
        return self.kernel

    def get_params(self):
        return [self.get_kernel()]

    def get_nparams(self):
        return {'kernel':self.kernel}

    def get_inputShape(self):
        return self.intputShape

    def get_outputShape(self):
        return (1, 1)

    def connect(self, *layers):
        assert len(layers) == 1
        self.intputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.init_kernel( self.get_inputShape()[1] ) #assume inputShape is 1D
        outputTensor = self.func(self)
        self.set_outputTensor( outputTensor )

class fixedRBFLayer(RBFLayer):
    '''
    Uses user defined kernel and during the training, the kernel are not changed
    However, the computation may report an mis-match error if user defined kernel with wrong shape
    '''
    def __init__(self, func, kernel):
        RBFLayer.__init__(self, func)
        self.kernel = kernel

    def init_kernel(self, numOfConnections):
        '''
        Ignore the input
        '''
        pass

    def get_params(self):
        return []

    def get_nparams(self):
        return dict()
