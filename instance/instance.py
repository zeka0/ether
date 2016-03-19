import numpy as np
import os
import theano
from theano import tensor as T
import warnings

from pool import *
from reader import *

class mixValueInterpreter(object):
    '''
    Mix interpreting the regression and classification
    '''
    def __init__(self, classIndices, regresIndices, classList):
        '''
        classIndices is for storing the indice of classification
        regresIndices is for storing the indice of regression
        '''
        self.classIndices = classIndices
        self.regresIndices = regresIndices
        self.targetInter = []
        for classes in classList:
            self.targetInter.append(targetInterpreter(classes))

    def get_numOf_attrs(self):
        num = 0
        for inter in self.targetInter:
            num += inter.get_numOf_attrs()
        num += len(self.regresIndices)
        return num

    def create_list(self, values):
        '''
        value must be iterable
        '''
        li = []
        classNum = 0
        for i in xrange(len(values)):
            if i in self.classIndices:
                li.extend(self.targetInter[classNum].create_list(values[i]))
            elif i in self.regresIndices:
                li.append(values[i])
            else: pass
        return li


    def create_tuple(self, values):
        return tuple(i for i in self.create_list(values))


class regressionInterpreter(object):
    def __init__(self):
        pass

    def create_tuple(self, value, isiteratble=False):
        if not isinstance(value, tuple):
            if isiteratble:
                return tuple(i for i in value)
            else: return value
        else: return value


class classificationInterpreter(object):
    '''
    Interprete strings into 0, 1 sequences presenting classes
    '''
    def __init__(self, attrList):
        if attrList is None or len(attrList)==0:
            raise interpreterException('classes provided isn\'t valid')
        self.attrList=attrList

    def get_numOf_attrs(self):
        '''
        Situation varies, so the super-class can't safely calculate the number of attributes
        the number of attributes can be considered as the number of input or output units
        '''
        raise NotImplementedError()

    def create_list(self, value):
        '''
        It assumes that the value has the same order as self.attrList
        '''
        raise NotImplementedError()

    def create_tuple(self, value):
        '''
        Public interface for interpreter
        When calling this function, one should make sure that the value and the attrList are in same order
        '''
        raise NotImplementedError()

'''
To use classificationInterpreters, the values to be interpreted should be classes
If there are multiple classes, the interpreter concatenates all of them into one tuple
'''
class multi_targetInterpreter(classificationInterpreter):
    #classList is a list of lists
    def __init__(self, classList):
        classificationInterpreter.__init__(self, classList)
        self.targetInters=[]
        for classes in classList:
            self.targetInters.append(targetInterpreter(classes))

    def create_list(self, values):
        li = []
        for inter, value in zip(self.targetInters, values):
            li.extend(inter.create_list(value))
        return li

    def create_tuple(self, values):
        return tuple(i for i in self.create_list(values))

    def get_numOf_attrs(self):
        num = 0
        for inter in self.targetInters:
            num += inter.get_numOf_attrs()
        return num

class targetInterpreter(classificationInterpreter):
    def __init__(self, classes):
        classificationInterpreter.__init__(self, classes)
        self.targetList=[]
        for i in xrange(self.get_numOf_attrs()):
            self.targetList.append(0)
        self.lastModifiedIndex=0

    def create_list(self, target):
        self.targetList[self.lastModifiedIndex]=0
        self.lastModifiedIndex=self.attrList.index(target)
        self.targetList[self.lastModifiedIndex]=1
        return self.targetList

    def create_tuple(self, target):
        return tuple(i for i in self.create_list(target))

    def get_numOf_attrs(self):
        return len(self.attrList)

class instance(object):
    '''
    instance is not responsible for shaping the vector
    '''
    def __init__(self, vecOfAttr, vecOfTarget):
        self.vecOfAttr=vecOfAttr
        self.vecOfTarget=vecOfTarget

    def get_attr(self):
        return self.vecOfAttr

    def get_target(self):
        return self.vecOfTarget

class dataBase:
    '''
    After the construction of dataBase
    You should neither call stochastic_init or full_init to finish construction
    You should always call "has_nextInstance" before call "get_nextInstance"
    Calling them with same-parameters
    Otherwise be-prepared to catch exceptions
    '''
    def __init__(self, reader, isStochastic, filter):
        self.reader = reader
        self.filter = filter
        self.isStochastic=isStochastic

    def set_state(self, isTrain):
        self.pool.set_state(isTrain)

    def __read_instances(self, batchSize):
        '''
        Filter the instances if needed
        '''
        instances=[]
        for ins in self.reader.read_instance(batchSize):
            if self.filter.is_valid(ins):
                if self.filter.should_filter(ins):
                    instances.append(self.filter.filter_instance(ins))
                else: instances.append(ins)
        return instances

    def stochastic_init(self, initInstanceNum=100):
        self.pool = stochastic_pool(self.__read_instances(initInstanceNum))

    def full_init(self, sampleRate=0.1, initInstanceNum=1000, isTrain=True):
        self.pool = full_pool(self.__read_instances(initInstanceNum), sampleRate, isTrain)

    def get_numOf_Attrs(self):
        return self.reader.get_numOf_Attrs()

    def get_numOf_Targets(self):
        return self.reader.get_numOf_Targets()

    def has_nextInstance(self, batchSize):
        if not self.isStochastic:
            return self.pool.has_nextInstance(batchSize)
        hasNext=self.pool.has_nextInstance(batchSize) #For stochastic pool, refill
        if hasNext: return True
        instances = self.__read_instances(self.pool.get_max_bufferSize()-self.pool.get_bufferSize())
        self.pool.set_instances(instances) #For stochastic pool
        return self.pool.has_nextInstance(batchSize)

    def get_nextInstance(self, batchSize=1):
        '''
        May throw instanceException
        '''
        return self.pool.get_nextInstance(batchSize)

    def get_bufferSize(self):
        return self.pool.get_bufferSize()

    def get_attrTar_tuple(self):
        return (self.reader.get_attrName(), self.reader.get_tarName())