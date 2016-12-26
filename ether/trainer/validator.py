import theano
from theano import tensor as T

from ether.trainer.core import validatorBase


def argmin(x):
    return T.argmin(x)

def argmax(x):
    return T.argmax(x)

class classifyValidator(validatorBase):
    def __init__(self, func, valid_print=True):
        #assert func in [argmax, argmin]
        self.func = func
        self.valid_print = valid_print

    def init_validation(self):
        predict = T.matrix()
        valid_cond = self.func(predict)#since the predict is a matrix, we should specify the dimension that should be used
        self.validation_function = theano.function(inputs=[predict], outputs=valid_cond) #argmax returns the index of the max value

    def diver_compute(self, predict, target):
        '''
        Test if predict and target are the same
        '''
        print 'Predict\n', predict
        index = self.validation_function(predict)#The returned value is a ndarray
        if self.valid_print:
            print 'Predicted Index', index
            print 'Target Index', target
        if isinstance(target, int):
            return target == index
        elif len(target.shape) == 1:
            return target[index] == 1
        else: return target[0][index] == 1

def gausi_distance(predict, target):
    return (T.abs_(predict-target)/target).mean()

class regressionValidator(validatorBase):
    def __init__(self, func):
        self.func = func

    def init_validation(self):
        predict=T.dvector('p')
        target=T.dvector('t')
        valid_cond=self.func(predict, target)
        self.validation_function=theano.function([predict, target], valid_cond)

    def diver_compute(self, predict, target, divergenceRate=0.05):
        '''
        Return if target and predict should be considered the same
        Assume predict and target are numpy ndarray
        '''
        value = self.validation_function(predict, target)
        if value < divergenceRate:
            return True
        else: return False
