from core import *

def argmin(x):
    return T.argmin(x)

def argmax(x):
    return T.argmax(x)

class classifyValidator(validator):
    def __init__(self, func, valid_print=True):
        validator.__init__(self)
        assert func in [argmax, argmin]
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
        if isinstance(target, int):
            return target == index
        else: return target[index] == 1