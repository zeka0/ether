from nnet.util import *
from theano import tensor as T

def merge_params(*params):
    paramList = []
    for param in params:
        if param is not None:
            paramList.append(param)
    return paramList

class layer(object):
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

    def get_inputShape(self):
        raise NotImplementedError()

    def get_outputShape(self):
        raise NotImplementedError()

    def get_params(self):
        '''
        Subclasses should override this method to provide the parameters required to update
        If the parameters are more than one, return a list
        If there are no parameters to update (i.e, no trainable parameters, return None instead)
        It's required to return a iterable object (eg, )
        '''
        return None

    def has_trainableParams(self):
        '''
        Returns whether the layer has trainable parameters or not
        '''
        return self.get_params() is not None

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
        self.set_inputTensor( T.matrix() )
        self.set_outputTensor( self.get_inputTensor() )
        self.inputShape = inputShape

    def get_params(self):
        return None

    def verify_shape(self):
        pass

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.get_inputShape()

    def connect(self, *layers):
        raise connectException('inputLayer can\'t connect to other layers')
