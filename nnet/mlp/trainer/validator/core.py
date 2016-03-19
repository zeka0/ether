from nnet.util.util import *

class validator(nnetController):
    def __init__(self):
        nnetController.__init__(self)

    def set_owner(self, nnet):
        '''
        To initialize the validation function
        '''
        nnetController.set_owner(self, nnet)
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
        predict = self.predict(attr)
        if not self.diver_compute(predict, tar):
            self.totalError = self.totalError +1

    def get_error_rate(self):
        return (float)(self.totalError) / (float)(self.totalNum)
