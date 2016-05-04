from ether.util import *
from layer import *

class nnet(object):
    '''
    Provide a general interface to components
    Unlike previously-designed nnet, this one makes it possible for outsiders to initialize the net for it
    Providing a better control over the layers
    It's considered as a great advantage because you may now pre-train layers with greater ease
    '''
    def __init__(self, dataBase, optimizer, validator, layers):
        self.set_layers( layers )
        self.dataBase=dataBase #Storage
        self.targetTensor=T.vector('T')
        self.optimizer=optimizer
        self.optimizer.set_owner(self)
        self.validator=validator
        self.validator.set_owner(self)

    def set_validator(self, validator):
        self.validator = validator
        self.validator.set_owner(self)

    def set_layers(self, layers):
        '''
        The layers must be pre-set before the nnet is put into use
        '''
        self.layers = layers

    def get_layers(self):
        return self.layers

    def get_params(self):
        '''
        Returns weights in ascending order of the layers
        Will be deprecated in the future
        '''
        params=[]
        for i in range(1, len(self.layers)): #Skip inputLayer
            if self.get_layers()[i].has_trainableParams():
                params.extend(self.layers[i].get_params()) #Add paras in specific-order
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
            self.outputFunction=theano.function(inputs=[self.get_inputTensor()], outputs=self.get_outputTensor())
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

    def verify_shape(self):
        for layer in self.get_layers():
            layer.verify_shape()

