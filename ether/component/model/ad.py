import theano
from theano import tensor as T
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams
from core import unsupervisedModel
from ether.component.init import init_shared, init_input

class autoEncoder(unsupervisedModel):
    def __init__(self, n_visible, n_hidden, **kwargs):
        pass
    def get_nparams(self):
        pass

    def get_params(self):
        pass

    def get_gparams(self):
        pass

    def get_monitoring_cost(self):
        pass

    def get_cost(self):
        pass

    def get_extra_updates(self):
        pass

    def get_inputShape(self):
        pass

    def get_inputTensor(self):
        pass

    def get_outputTensor(self):
        pass

    def get_outputShape(self):
        pass
