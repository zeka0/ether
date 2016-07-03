from ether.component.layer import *
from ether.component.model.controller import controller
from theano import tensor as T
import theano

class optimizer(controller):
    def set_owner(self, model):
        '''
        Call controller.set_owner(self, model)
        '''
        controller.set_owner(self, model)
        self.init_train()

    def init_train(self):
        '''
        Called after the owner is set
        '''
        raise NotImplementedError()

    def train_once(self, attr, tar):
        raise NotImplementedError()

class optimizerBase(optimizer):
    def get_updates(self):
        '''
        :return updates for trainable parameters
        :rtype dict
        '''
        raise NotImplementedError()

    def add_additionalOutputs(self, *outputs):
        '''
        Add outputs to the outputs_list of the theano.function
        '''
        if not hasattr(self, 'outputList'):
            self.outputList = [out for out in outputs]
        else:
            self.outputList.extend(outputs)

    def get_additionalOutputs(self):
        return self.outputList

    def init_train(self):
        if self.is_supervise():
            inputs = [self.get_inputTensor(), self.get_targetTensor()]
        else: inputs = [self.get_inputTensor()]
        if hasattr(self, 'outputList'):
            self.train=theano.function(
                inputs=inputs,
                outputs=self.get_additionalOutputs(),
                updates=self.get_updates()
            )
        else:
            self.train=theano.function(
                inputs=inputs,
                outputs=[],
                updates=self.get_updates()
            )

    def train_once(self, attr, tar):
        try:
            if self.is_supervise():
                return self.train(attr, tar) #Discard output
            else: return self.train(attr)
        except instanceException:
            #no more instances available
            print 'exception occured in instance availability'


class validatorBase(controller):
    def set_owner(self, model):
        '''
        To initialize the validation function
        '''
        controller.set_owner(self, model)
        self.init_validation()
        self.totalNum = 0
        self.totalError = 0

    def init_validation(self):
        '''
        initialize the validation function
        '''
        raise NotImplementedError()

    def diver_compute(self, predict, tar):
        '''
        return the validation function
        '''
        raise NotImplementedError()

    def validate_once(self, attr, tar):
        self.totalNum = self.totalNum +1
        predict = self.feed_forward(attr)
        if not self.diver_compute(predict, tar):
            self.totalError = self.totalError +1

    def get_error_rate(self):
        return (float)(self.totalError) / (float)(self.totalNum)
