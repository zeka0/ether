from core import layer
from ether.component.model.util.shape import dimShuffle_shape

class dimShuffelLayer(layer):
    def __init__(self, *pattern):
        layer.__init__(self)
        self.pattern = pattern

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.outputShape

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_inputShape()
        self.set_inputTensor( layers[0].get_outputTensor() )
        outputTensor = self.get_inputTensor().dimshuffle(*self.pattern)
        self.set_outputTensor(outputTensor)
        self.outputShape = dimShuffle_shape(self.inputShape, *self.pattern)

    def get_params(self):
        return []

    def get_nparams(self):
        return dict()