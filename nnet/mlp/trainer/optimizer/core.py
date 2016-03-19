from nnet.mlp.layer import *

class optimizerBase(nnetController):
    def __init__(self):
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
        '''
        raise NotImplementedError()

    def get_loss(self):
        '''
        return the tensor of the loss function
        '''
        raise NotImplementedError()

    def get_updates(self):
        '''
        Return a list of tuple of (para, update)
        '''
        raise NotImplementedError()

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
        self.train=theano.function(
            inputs=[self.get_inputTensor(), self.get_targetTensor()],
            outputs=[self.loss],
            updates=self.get_updates()
        )

    def train_once(self, attr, tar):
        try:
            self.train(attr, tar) #Discard output
        except instanceException:
            #no more instances available
            print 'exception occured in instance availability'
