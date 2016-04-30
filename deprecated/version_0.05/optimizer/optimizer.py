from core import *
from ether.component.initialize import init_shared

'''
When using this module, one thing to bear in mind is that you should check if loss function uses the targetTensor
If not, this will raise the theano.compile.funcion_module.UnusedInputError
To disable it, add 'on_unused_input = 'warn'' to the theano.function
'''

class SGDOptimizer(optimizerBase):
    def __init__(self, func, learningRate=0.01):
        optimizerBase.__init__(self, func)
        self.learningRate = learningRate

    def get_learningRate(self):
        return self.learningRate

    def get_updates(self):
        gradParams = self.get_gradients()
        updatesList = []
        for gpTuple in gradParams:
            grad = gpTuple[0]
            para = gpTuple[1]
            update = para - grad * self.learningRate
            updatesList.append( (para, update) ) #Stochastic batch
        return updatesList

class adaGradOptimizer(optimizerBase):
    def __init__(self, func, learningRate=0.01):
        optimizerBase.__init__(self, func)
        self.learningRate = learningRate

    def get_learningRate(self):
        return self.learningRate

    def get_updates(self):
        gradParaTuples = self.get_gradients()
        accumulators = [theano.shared( np.zeros(p.get_value().shape) ) for g, p in gradParaTuples]
        updatesList = []

        for gpTuple, acc in zip(gradParaTuples, accumulators):
            grad = gpTuple[0]
            para = gpTuple[1]
            new_acc = acc + grad ** 2  # update accumulator
            updatesList.append((acc, new_acc))
            new_param = para - self.learningRate * grad / T.sqrt(new_acc)
            updatesList.append((para, new_param))  # apply constraints
        return updatesList
