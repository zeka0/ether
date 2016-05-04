from ether.util import *
from ether.component.init import *
from theano import tensor as T
from ether.component.core import component

def merge_params(*params):
    paramList = []
    for param in params:
        if param is not None:
            paramList.append(param)
    return paramList

class layer(component):
    '''
    It's required that the inputShape & outputShape be ndarray
    base class for all kinds of layers
    '''
    def __init__(self):
        pass

    def set_inputTensor(self, inputTensor):
        '''
        Subclass call this method to set the tensor
        '''
        self.inputTensor = inputTensor

    def set_outputTensor(self, outputTensor):
        '''
        Subclass call this method to set the tensor
        '''
        self.outputTensor = outputTensor

    def get_inputTensor(self):
        '''
        Instead of calling self.inputTensor directly, it's encouraged to call this method instead
        '''
        if hasattr(self, 'inputTensor'):
            return self.inputTensor
        else:
            raise connectException('connect before get inputTensor')

    def get_outputTensor(self):
        '''
        Instead of calling self.inputTensor directly, it's encouraged to call this method instead
        '''
        if hasattr(self, 'outputTensor'):
            return self.outputTensor
        else:
            raise connectException('connect before get outputTensor')

    def has_trainableParams(self):
        '''
        Returns whether the layer has trainable parameters or not
        '''
        return len(self.get_params()) != 0

    def connect(self, *layers):
        '''
        One layer may connect to multiple layers
        '''
        raise NotImplementedError()

class inputLayer(layer):
    '''
    A dummy layer for input
    '''
    def __init__(self, inputShape):
        layer.__init__(self)
        self.init_input(inputShape)

    def init_input(self, inputShape):
        self.set_inputTensor( init_input(inputShape) )
        self.set_outputTensor( self.get_inputTensor() )
        self.inputShape = inputShape

    def get_params(self):
        return []

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.get_inputShape()

    def connect(self, *layers):
        raise connectException('inputLayer can\'t connect to other layers')

class biasLayer(layer):
    def __init__(self, **biasKwargs):
        layer.__init__(self)
        self.biasKwargs = biasKwargs
        self.b = init_shared(shape=(1,), **self.biasKwargs)

    def connect(self, *layers):
        assert len(layers) == 1
        self.set_inputTensor(layers[0].get_outputTensor())
        self.set_outputTensor( self.get_inputTensor() + self.b )
        self.inputShape = layers[0].get_outputShape()

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.get_inputShape()

    def get_params(self):
        return [self.b]