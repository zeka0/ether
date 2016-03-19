__author__ = 'mac'

import theano
from theano import tensor as T
import numpy as np
from util import nnetController
from exception import *

def reshape_predict(predict):
    '''
    reshape the tensor if necessary
    '''
    if isinstance(predict, list):
        predict = predict[0]
    if len(predict.shape) >1 :
        return predict.reshape((predict.shape[0]*predict.shape[1],))
    else: return predict

class validator(nnetController):
    def __init__(self):
        nnetController.__init__(self)

    def set_owner(self, nnet):
        '''
        To initialize the validation function
        '''
        nnetController.set_owner(self, nnet)
        self.init_validation()

    def init_validation(self):
        '''
        initialize the validation function
        '''
        raise NotImplementedError()

    def diver_compute(self):
        '''
        return the validation function
        '''
        raise NotImplementedError()

    def validate(self, maxCycles):
        self.set_state(False) #Turn to validating mode
        totalNum = 0
        totalError = 0
        counter = 0
        while self.has_nextInstance() and counter < maxCycles:
            totalNum = totalNum +1
            instanceList = self.get_nextInstance()
            tars = self.predict(instanceList[0].get_attr())
            if not self.diver_compute(tars, instanceList[0].get_target()):
                totalError = totalError +1
            counter+=1
        return float(totalError)/float(totalNum)

class classifyValidator(validator):
    def __init__(self, tarInter):
        validator.__init__(self)
        self.tarInter = tarInter

    def init_validation(self):
        predict = T.dvector('p')
        valid_cond = T.argmax(predict)
        self.validation_function = theano.function([predict], valid_cond)

    def diver_compute(self, predict, target):
        '''
        diversity-compute
        Return if target and predict should be considered the same
        Assume predict and target are numpy ndarray
        '''
        predict = reshape_predict(predict)
        maxIndex = self.validation_function(predict)
        if target[maxIndex] == 1:
            return True
        else: return False

#TODO:test regressionValidator
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
