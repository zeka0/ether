import theano
from theano import tensor as T
import numpy as np
from core import supervisedModel, unsupervisedModel

class controller:
    def set_owner(self, model):
        '''
        Set the owner of this optimizer
        Called by model automatically
        '''
        self.model = model

    def get_owner(self):
        return self.model

    def get_inputTensor(self):
        return self.model.get_inputTensor()

    def get_outputTensor(self):
        return self.model.get_outputTensor()

    def get_params(self):
        return self.model.get_params()

    def get_gparams(self):
        return self.model.get_gparams()

    def get_targetTensor(self):
        '''
        Note that this method only works for the nnet
        '''
        return self.model.get_targetTensor()

    def feed_forward(self, input):
        return self.get_outputFunction()(input)

    def get_cost(self):
        return self.model.get_cost()

    def get_monitoring_cost(self):
        return self.model.get_monitoring_cost()

    def get_extra_updates(self):
        return self.model.get_extra_updates()

    def get_outputFunction(self):
        if not hasattr(self, 'outputFunction'):
            self.outputFunction=theano.function(inputs=[self.get_inputTensor()], outputs=self.get_outputTensor())
        return self.outputFunction

    def is_supervise(self):
        if isinstance(self.model, supervisedModel):
            return True
        else: return False