from core import *
class regressionValidator(validator):
    def __init__(self):
        validator.__init__(self)

    def init_validation(self):
        predict=T.dvector('p')
        target=T.dvector('t')
        valid_cond=(T.abs_(predict-target)/target).mean()
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
