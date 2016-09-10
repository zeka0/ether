from core import *
from ether.util.shape import *
from ether.component.init import init_shared
from ether.util.activation import *

#TODO check the dimension of the inputs
class recurrentLayer(layer):
    '''
    We can view weight-layer as a conv1D layer
    '''
    def __init__(self, numOfHUnits, hiddenToHiddenFn=tanh, **kwargs):
        layer.__init__(self)
        self.numHUnits = numOfHUnits
        self.hiddenToHiddenFn = hiddenToHiddenFn #The activate function for hidden to hidden connection
        assert kwargs.has_key('U')
        assert kwargs.has_key('V')
        assert kwargs.has_key('W')
        self.UKwargs = kwargs['U']
        self.VKwargs = kwargs['V']
        self.WKwargs = kwargs['W']
        assert not self.UKwargs.has_key('shape')
        assert not self.VKwargs.has_key('shape')
        assert not self.WKwargs.has_key('shape')

    def init_hiddenToVisible(self):
        #Init V
        self.V= init_shared(shape=(self.get_inputShape()[0], self.numHUnits), **self.VKwargs)

    def init_visibleToHidden(self):
        #Init U
        #TODO may have bug in shape calculation
        self.U= init_shared(shape=(self.numHUnits, self.get_inputShape()[0]), **self.UKwargs)

    def init_hiddenToHidden(self):
        #Init W
        self.W= init_shared(shape=(self.numHUnits, self.numHUnits,), **self.WKwargs)

    def get_hiddenToVisible(self):
        return self.V

    def get_hiddenToHidden(self):
        return self.W

    def get_visibleToHidden(self):
        return self.U

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.get_inputShape()

    def get_params(self):
        return [self.U, self.V, self.W]

    def get_nparams(self):
        return {'U':self.U, 'V':self.V, 'W':self.W}


    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        assert len(self.inputShape) == 2
        self.set_inputTensor( layers[0].get_outputTensor() )

        self.init_visibleToHidden()
        self.init_hiddenToHidden()
        self.init_hiddenToVisible()
        #TODO may have bug in U.dot(x_t) and return[..., o_t]
        def forward_prop_step(x_t, s_t_prev, U, V, W):
            s_t = self.hiddenToHiddenFn(U.dot(x_t) + W.dot(s_t_prev))
            o_t = V.dot(s_t)
            return [s_t, o_t]

        [s_t, o_t], updates = theano.scan(fn=forward_prop_step,
                                          outputs_info=[None, np.zeros((self.numHUnits, self.numHUnits))],
                                          sequences=self.get_inputTensor(),
                                          non_sequences=[self.U, self.V, self.W])
        outputTensor = o_t
        self.set_outputTensor(outputTensor)
