from ether.component.layer import *
from ether.util.controller import *

class optimizerBase(nnetController):
    def __init__(self, func):
        '''
        func is a function object, which will acceptr an optimizer as its parameter and
        return a tensor as the loss
        '''
        self.func = func
        nnetController.__init__(self)

    def set_owner(self, nnet):
        '''
        To initialize the cost function
        '''
        nnetController.set_owner(self, nnet)
        self.init_loss()
        self.init_train()

    def init_loss(self):
        '''
        call init_loss to initialize the loss function
        this method can't be called before the set_owner method
        '''
        self.loss = self.func(self)

    def get_loss(self):
        '''
        return the tensor of the loss function
        '''
        return self.loss

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

    def get_gradients(self):
        '''
        Return a list of tuple
        Tuple of shape (grad, parameter)
        '''
        if not hasattr(self, 'gradParams'):
            self.gradParams = []
            grads = T.grad(self.get_loss(), self.get_params())
            for grad, para in zip(grads, self.get_params()):
                self.gradParams.append((grad, para))
        return self.gradParams

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