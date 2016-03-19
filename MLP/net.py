#Import of foreign modules
import theano
from theano import tensor as T
import numpy as np

#Import of self-designed modules
import layer
from theanoNnet.nnetUtil import util
from theanoNnet.nnetUtil.exception import *

class nnet(object):
    '''
    Provide a general interface to components
    '''
    def __init__(self, hiddenLayerUnitsNums,
                 hiddenLayerActFuncs, inputFunc, outputFunc,
                 dataBase, optimizer, validator):
        instanceAttrNum = dataBase.get_numOf_Attrs()
        targetNum = dataBase.get_numOf_Targets()
        self.dataBase=dataBase #Storage
        self.init_layers(instanceAttrNum, targetNum,
                         hiddenLayerUnitsNums, hiddenLayerActFuncs, inputFunc, outputFunc)
        self.targetTensor=T.dvector('T')
        self.optimizer=optimizer
        self.optimizer.set_owner(self)
        self.validator=validator
        self.validator.set_owner(self)

    def init_layers(self, instanceAttrNum, targetNum,
                    hiddenLayerUnitsNums, hiddenLayerActFuncs, inputFunc, outputFunc):
        '''
        Subclass may override this
        User shouldn't call this directly
        '''
        self.layers=[]
        preLayer=layer.inputLayer(instanceAttrNum, inputFunc, util.initBias(instanceAttrNum), 0)
        self.layers.append(preLayer) #Append inputLayer

        for unitNum, actFunc, i in zip(hiddenLayerUnitsNums, hiddenLayerActFuncs,
                                       xrange(len(hiddenLayerUnitsNums))):
            self.layers.append(layer.layer(unitNum, actFunc, util.initBias(unitNum), idNum=i))
            self.layers[-1].connect(preLayer)
            preLayer=self.layers[-1]

        self.layers.append(layer.outputLayer(targetNum, outputFunc, len(hiddenLayerUnitsNums)+1))
        self.layers[-1].connect(preLayer)

    def get_params(self):
        '''
        returns weights in ascending order of the layers
        '''
        params=[]
        for i in range(1, len(self.layers)): #Skip inputLayer
            params.append(self.layers[i].get_weights()) #Add paras in specific-order
        return params

    def get_nextInstance(self, batchSize=1):
        '''
        Return target vec
        '''
        if self.dataBase.has_nextInstance(batchSize):
            return self.dataBase.get_nextInstance(batchSize)
        else: raise instanceException('nnet has no more instances')

    def predict(self, attrVec):
        return self.cal_output(attrVec)

    def cal_output(self, attrVec):
        outputFunc = self.get_outputFunction()
        predict = outputFunc(attrVec)
        return predict #reshaping prediction

    def get_inputTensor(self):
        return self.layers[0].get_inputTensor()

    def get_outputTensor(self):
        return self.layers[-1].get_outputTensor()

    def get_targetTensor(self):
        return self.targetTensor

    def get_outputFunction(self):
        if not hasattr(self, 'outputFunction'):
            self.outputFunction=theano.function([self.get_inputTensor()], self.get_outputTensor())
        return self.outputFunction

    def get_layerOutputTensors(self):
        li = []
        for layer in self.layers:
            li.append(layer.get_outputTensor())
        return li

    def has_nextInstance(self, batchSize=1):
        return self.dataBase.has_nextInstance(batchSize)

    def set_state(self, isTrain):
        self.dataBase.set_state(isTrain)

