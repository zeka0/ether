from core import *
from ether.util.shape import *
from ether.component.init import init_shared

class weightLayer(layer):
    '''
    We can view weight-layer as a conv1D layer
    '''
    def __init__(self, numOfOutput, **kwargs):
        layer.__init__(self)
        self.numOfOutput = numOfOutput
        assert kwargs.has_key('weight')
        assert kwargs.has_key('bias')
        self.weightKwargs = kwargs['weight']
        self.biasKwargs = kwargs['bias']
        assert not self.weightKwargs.has_key('shape')
        self.init_bias()

    def init_weights(self):
        self.W = init_shared(shape=self.get_weightShape(), **self.weightKwargs)

    def init_bias(self):
        self.b = init_shared(shape=(self.numOfOutput,), **self.biasKwargs)

    def get_weights(self):
        return self.W

    def get_bias(self):
        return self.b

    def get_inputShape(self):
        return self.inputShape

    def get_weightShape(self):
        return (self.get_inputShape()[1], self.numOfOutput)

    def get_outputShape(self):
        return ( self.inputShape[0], self.numOfOutput )

    def get_params(self):
        return [self.W, self.b]

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        assert len(self.inputShape) == 2
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.init_weights()
        outputTensor = self.get_inputTensor().dot( self.W ) + self.b
        self.set_outputTensor(outputTensor)
