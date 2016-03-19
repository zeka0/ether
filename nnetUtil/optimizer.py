__author__ = 'mac'

import theano
from theano import tensor as T
import numpy as np
from util import nnetController
from exception import *

isVerbose = True

class optimizer(nnetController):
    def __init__(self):
        nnetController.__init__(self)

    def set_owner(self, nnet):
        '''
        To initialize the cost function
        '''
        nnetController.set_owner(self, nnet)
        self.init_loss()

    def train(self):
        '''
        Be sure to call self.set_state(True) before training
        call before training self.set_state(True)
        '''
        raise NotImplementedError()

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

    def get_gradients(self):
        '''
        use subgraph_grad to perform back-propogate
        '''
        weights = self.get_params() #weights in ascending order of layers
        outs = self.get_layerOutputTensors()
        next_grad = None
        param_grads = []

        params = [[weight] for weight in weights]
        params.reverse() #reverse to descending order
        grad_ends = [[out] for out in outs]
        grad_ends.pop() #pop out outputLayer outs
        grad_ends.reverse() #reverse to descending order
        cost = self.get_loss() #TODO:add regularizers here
        for i in xrange(len(weights)):
            param_grad, next_grad = theano.subgraph_grad(
                wrt=params[i], end=grad_ends[i], #skip the outputLayer
                start=next_grad, cost=cost
            )
            next_grad = dict(zip(grad_ends[i], next_grad))
            param_grads.extend(param_grad)
            cost = None #no cost after outputLayer
        param_grads.reverse() #reverse to ascending order
        return param_grads

class sqrtMeanOptimizer(optimizer):
    def __init__(self):
        optimizer.__init__(self)

    def init_loss(self):
        self.loss = T.sqrt(self.get_targetTensor()-self.get_outputTensor()).sum() #Square error

    def get_loss(self):
        return self.loss

    def train(self, cycles, learningRate=0.1):
        self.set_state(True)

        grads = self.get_gradients()
        params = self.get_params()
        updatesList = []
        for grad, para in zip(grads, params):
            updatesList.append((para, para-grad*learningRate)) #Stochastic batch
        train=theano.function(
            inputs=[self.get_inputTensor(), self.get_targetTensor()],
            outputs=[self.loss],
            updates=updatesList
        )
        for i in xrange(cycles):
            if self.has_nextInstance():
                try:
                    instanceList= self.get_nextInstance()
                    train(instanceList[0].get_attr(), instanceList[0].get_target()) #Discard output
                except instanceException:
                    #no more instances available
                    break
                if isVerbose:
                    print 'iteration', i, 'weights:'
                    for para in params:
                        print(para.get_value())
            else: break #stop training
