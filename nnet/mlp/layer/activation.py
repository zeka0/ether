from core import *
from nnet.util import *

class activationLayer(layer):
    '''
    Trivial class
    Any subclass that contains a activation function should sub-class this class
    '''
    def get_params(self):
        return None

class sigmoidLayer(activationLayer):
    def __init__(self):
        activationLayer.__init__(self)

    def connect(self, *layers):
        if len(layers) > 1:
            raise connectException('sigmoid layer can only connect one layer')
        self.outputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor )
        self.set_outputTensor( sigmoid( self.get_inputTensor() ) )

    def get_inputShape(self):
        return self.get_outputShape()

    def get_outputShape(self):
        return self.outputShape

    def verify_shape(self):
        '''
        sigmoid activation doesn't change the shape
        '''
        if self.get_inputShape() != self.get_outputShape():
            raise shapeError(self)

class tanhLayer(activationLayer):
    '''
    Tanh function follows the form A*tanh( S*input )
    '''
    def __init__(self, A, S):
        activationLayer.__init__(self)
        self.A = A
        self.S = S

    def connect(self, *layers):
        assert len(layers) == 1
        self.outputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.set_outputTensor( self.A * T.tanh( self.S * self.get_inputTensor() ) )

    def verify_shape(self):
        '''
        sigmoid activation doesn't change the shape
        '''
        if self.get_inputShape() != self.get_outputShape():
            raise shapeError(self)

    def get_inputShape(self):
        return self.get_outputShape()

    def get_outputShape(self):
        return self.outputShape

class argmaxLayer(activationLayer):
    def __init__(self):
        activationLayer.__init__(self)

    def connect(self, *layers):
        assert len(layers) == 1
        self.outputShape = (1, 1)
        self.inputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.set_outputTensor( argmax( self.get_inputTensor() ) )

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.outputShape

    def verify_shape(self):
        pass

class softmaxLayer(activationLayer):
    def __init__(self):
        activationLayer.__init__(self)

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.set_outputTensor( softmax( self.get_inputTensor() ) )

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.get_inputShape()

    def verify_shape(self):
        pass

class minusLayer(activationLayer):
    def __init__(self):
        activationLayer.__init__(self)

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.set_outputTensor( -( self.get_inputTensor() ) )

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.get_inputShape()

    def verify_shape(self):
        pass
