#Import of foreign modules
import numpy as np
import theano
from theano import tensor as T

from nnet.util import util

'''
The Layer is modeled after a graph
The loss function and the output function doen's require any input tensor
The so-called input-tensor is set using a shared variable
'''

class layer:
    '''
    Class for storing units and states
    '''
    def __init__(self, numOfUnits, actFunc, bias, idNum):
        self.numOfUnits=numOfUnits
        self.actFunc=actFunc
        self.bias=bias
        self.idNum = idNum

    def __generate_weightName(self, preLayer):
        return 'W'+str(preLayer.idNum)+str(self.idNum)

    def connect(self, anoLayer):
        '''
        Connects this to the anoLayer
        This is connected above the anoLayer(i.e, anoLayer's outputs becomes this inputs)
        '''
        self.preLayer=anoLayer
        self.weights=theano.shared(
            util.initWeightMatrix(self.preLayer.numOfUnits, self.numOfUnits),
            self.__generate_weightName(anoLayer)) #Sharing Weights
        inPuts=self.preLayer.get_outputTensor() #Connects graphs
        self.outputTensor=self.actFunc(inPuts.dot(self.weights)+ self.bias)

    def get_outputTensor(self):
        return self.outputTensor

    def get_weights(self):
        return self.weights

class inputLayer(layer):
    '''
    Assume the numOfUnits is the same of instanceAttrNum
    '''
    def __init__(self, numOfUnits, actFunc, bias, idNum):
        layer.__init__(self, numOfUnits, actFunc, bias, idNum)
        self.inputTensor=T.dvector('In') #Define of inputLayer.outputTensor

    def get_inputTensor(self):
        return self.inputTensor

    def get_outputTensor(self):
        return self.get_inputTensor()

    def connect(set, anoLayer):
        raise NotImplementedError('inputLayer can\'t connect other layers')

class outputLayer(layer):
    def __init__(self, numOfUnits, actFunc, idNum):
        layer.__init__(self, numOfUnits, actFunc, np.array(0), idNum) #No bias
