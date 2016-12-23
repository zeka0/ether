from core import *
from collections import OrderedDict
'''
When using this module, one thing to bear in mind is that you should check if loss function uses the targetTensor
If not, this will raise the theano.compile.funcion_module.UnusedInputError
To disable it, add 'on_unused_input = 'warn'' to the theano.function
'''

class SGDOptimizer(optimizerBase):
    def __init__(self, learningRate=0.01):
        self.learningRate = learningRate

    def get_learningRate(self):
        return self.learningRate

    def get_updates(self):
        gradParams = self.get_gparams()
        updateDict = OrderedDict()
        for gpTuple in gradParams:
            grad = gpTuple[0]
            para = gpTuple[1]
            #TODO bug
            update = para - grad * self.learningRate #take the mean of the batch
            updateDict[para] = update #Stochastic batch
        updateDict.update( self.get_extra_updates() )
        return updateDict

class AdaGradOptimizer(optimizerBase):
    def __init__(self, learningRate=0.01):
        self.learningRate = learningRate

    def get_learningRate(self):
        return self.learningRate

    def get_updates(self):
        gradParaTuples = self.get_gparams()
        accumulators = [theano.shared( np.zeros(p.get_value().shape) ) for g, p in gradParaTuples]
        updateDict = OrderedDict()

        for gpTuple, acc in zip(gradParaTuples, accumulators):
            grad = gpTuple[0]
            para = gpTuple[1]
            new_acc = acc + grad ** 2  # update accumulator
            updateDict[acc] = new_acc
            new_param = para - self.learningRate * T.mean(grad) / T.sqrt(new_acc)
            updateDict[para] = new_param
        updateDict.update( self.get_extra_updates() )
        return updateDict
