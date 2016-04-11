from core import *

def gausi_distance(predict, target):
    return (T.abs_(predict-target)/target).mean()


class regressionValidator(validator):
    def __init__(self, func):
        validator.__init__(self)
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
