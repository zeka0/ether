import theano
from theano import tensor as T
import numpy as np
from core import unsupervisedModel
from ether.component.init import init_shared, init_input, theano_rng

class autoEncoder(unsupervisedModel):
    def __init__(self, n_visible, n_hidden, **kwargs):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        if not isinstance(kwargs['weight'], dict):
            self.weight = kwargs['weight']
        else:
            assert not kwargs['weight'].has_key('shape')
            self.weight = init_shared(shape=(n_visible, n_hidden), **kwargs['weight'] )

        if not isinstance(kwargs['vbias'], dict):
            self.vbias = kwargs['vbias']
        else:
            assert not kwargs['vbias'].has_key('shape')
            self.vbias = init_shared(shape=(n_visible,), **kwargs['vbias'] )

        if not isinstance(kwargs['hbias'], dict):
            self.hbias = kwargs['hbias']
        else:
            assert not kwargs['hbias'].has_key('shape')
            self.hbias = init_shared(shape=(n_hidden,), **kwargs['hbias'] )

        self.theano_rng = theano_rng
        self.input = T.matrix()
        self.weight_prime = self.weight.T

    def get_nparams(self):
        nparam = dict()
        nparam['weight'] = self.weight
        nparam['vbias'] = self.vbias
        nparam['hbias'] = self.hbias
        return nparam

    def get_params(self):
        return [self.hbias, self.vbias, self.weight]

    def get_gparams(self):
        cost = self.get_cost()
        gparams = T.grad(cost, self.get_params())
        gpts = []
        for gparam, para in zip(gparams, self.get_params()):
            gpts.append((gparam, para))
        return gpts

    def get_monitoring_cost(self):
        return self.get_cost()

    def get_cost(self):
        z = self.get_outputTensor()
        L = - T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - self.input), axis=1)
        cost = T.mean(L)
        return cost

    def get_extra_updates(self):
        return []

    def get_inputShape(self):
        return (1, self.n_visible)

    def get_inputTensor(self):
        return self.input

    def get_outputTensor(self):
        y = self.get_hidden_values(self.input)
        z = self.get_reconstructed_input(y)


    def get_outputShape(self):
        return self.get_inputShape()

    def get_hidden_values(self, input):
        '''
        Computes the values of the hidden layer
        '''
        return T.nnet.sigmoid(T.dot(input, self.weight) + self.hbias)

    def get_reconstructed_input(self, hidden):
        '''
        Computes the reconstructed input given the values of the hidden layer
        '''
        return T.nnet.sigmoid(T.dot(hidden, self.weight_prime) + self.vbias)
