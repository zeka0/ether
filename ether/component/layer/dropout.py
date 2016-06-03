from core import layer
from ether.component.init import theano_rng
import theano
import numpy as np

class dropoutLayer(layer):
    '''
    It's required that the inputShape & outputShape be ndarray
    base class for all kinds of layers
    '''
    def __init__(self, app_p):
        '''
        :param app_p: possibility of appear of the output of previous layer
        app_p is also known as the possibility of 1
        :type app_p: float
        '''
        self.app_p = app_p

    def connect(self, *layers):
        assert len(layers) == 1
        self.set_inputTensor( layers[0].get_outputTensor() )
        self.inputShape = layers[0].get_outputShape()

        self.bitvec_tensor = theano_rng.binomial(size=self.get_inputShape(), p=self.app_p, dtype=theano.config.floatX)
        self.rngfun = theano.function(inputs=[], outputs=self.bitvec_tensor)

        '''
        Deliberately make bitvec to be a shared variable
        To make later operation of reset it possible
        '''
        self.bitvec = theano.shared( self.rngfun() )
        outputTensor = self.get_inputTensor() * self.bitvec
        self.set_outputTensor( outputTensor=outputTensor )

    def reset_bitvec(self):
        '''
        Reset the bitvec randomly
        '''
        self.bitvec.set_value(self.rngfun())

    def cancel_bitvec(self):
        '''
        Cancel the effect of the randomness of bitvec
        This will make the output of the dropout to be averanged by a factor of appear-possibility
        '''
        self.bitvec.set_value( self.app_p * np.ones(self.get_inputShape()) )

    def get_params(self):
        return []

    def get_nparams(self):
        return dict()

    def get_inputShape(self):
        '''
        :return the shape of the input
        :rtype tuple
        '''
        return self.inputShape

    def get_outputShape(self):
        '''
        :return the shape of the output
        :rtype tuple
        '''
        return self.get_inputShape()
