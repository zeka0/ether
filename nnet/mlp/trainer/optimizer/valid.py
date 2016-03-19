from algorithnm import *

class sqrtMean_SGDOptimizer(SGDOptimizer):
    def __init__(self):
        SGDOptimizer.__init__(self)

    def init_loss(self):
        self.loss = T.sqrt( self.get_targetTensor() - self.get_outputTensor() ).sum()

class crossEntro_SGDOptimizer(SGDOptimizer):
    def __init__(self):
        SGDOptimizer.__init__(self)

    def init_loss(self):
        self.loss = - ( self.get_targetTensor().dot( ( T.log( self.get_outputTensor() / self.get_targetTensor() ) ).T ) ).sum()

class lenet_SGDOptimizer(SGDOptimizer):
    def __init__(self):
        SGDOptimizer.__init__(self)

    def init_loss(self):
        self.loss = self.get_outputTensor()[0][ T.argmax(self.get_targetTensor()) ] + T.log( 6 + ( T.exp( - self.get_outputTensor() ) ).sum() )

class crossEntro_AdaOptimizer(adaGradOptimizer):
    def __init__(self):
        adaGradOptimizer.__init__(self)

    def init_loss(self):
        self.loss = - ( self.get_targetTensor().dot( ( T.log( self.get_outputTensor() / self.get_targetTensor() ) ).T ) ).sum()

class lenet_AdaOptimizer(adaGradOptimizer):
    def __init__(self):
        adaGradOptimizer.__init__(self)

    def init_loss(self):
        self.loss = self.get_outputTensor()[0][ T.argmax(self.get_targetTensor()) ] + T.log( 6 + ( T.exp( - self.get_outputTensor() ) ).sum() )
