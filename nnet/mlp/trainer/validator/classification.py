from core import *
class classifyValidator(validator):
    def __init__(self):
        validator.__init__(self)

    def init_validation(self):
        predict = T.matrix()
        valid_cond = T.argmax(predict)#since the predict is a matrix, we should specify the dimension that should be used
        self.validation_function = theano.function(inputs=[predict], outputs=valid_cond) #argmax returns the index of the max value

    def diver_compute(self, predict, target):
        '''
        Test if predict and target are the same
        '''
        maxIndex = self.validation_function(predict)#The returned value is a ndarray
        print 'Predicted Index', maxIndex
        if isinstance(target, int):
            return target == maxIndex
        else: return target[maxIndex] == 1
