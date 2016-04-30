from ether.component.layer import *
from ether.component.model.controller import controller
from theano import tensor as T
import theano

class optimizerBase(controller):
    def set_owner(self, nnet):
        '''
        To initialize the cost function
        '''
        controller.set_owner(self, nnet)
        self.init_train()

    def get_updates(self):
        '''
        Return a list of tuple of (para, update)
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
        if hasattr(self, 'outputList'):
            self.train=theano.function(
                inputs=[self.get_inputTensor(), self.get_targetTensor()],
                outputs=self.get_additionalOutputs(),
                updates=self.get_updates()
            )
        else:
            self.train=theano.function(
                inputs=[self.get_inputTensor(), self.get_targetTensor()],
                outputs=[],
                updates=self.get_updates()
            )

    def train_once(self, attr, tar):
        try:
            return self.train(attr, tar) #Discard output
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
