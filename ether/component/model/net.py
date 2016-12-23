from ether.component.layer import *
from core import supervisedModel

class nnet(supervisedModel):
    '''
    Provide a general interface to components
    Unlike previously-designed nnet, this one makes it possible for outsiders to initialize the net for it
    Providing a better control over the layers
    It's considered as a great advantage because you may now pre-train layers with greater ease
    '''
    def __init__(self, layers, cost_func, monitor_cost_func):
        self.set_layers( layers )
        self.targetTensor=T.matrix()
        self.cost_func = cost_func
        self.monitor_cost_func = monitor_cost_func

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

    def get_nparams(self):
        raise NotImplementedError('Net doesn\'t support nparams\n' )

    def get_inputTensor(self):
        return self.layers[0].get_inputTensor()

    def get_outputTensor(self):
        return self.layers[-1].get_outputTensor()

    def get_targetTensor(self):
        return self.targetTensor

    def get_layerOutputTensors(self):
        li = []
        for layer in self.layers:
            li.append(layer.get_outputTensor())
        return li

    def get_cost(self):
        return self.cost_func(self)

    def get_monitoring_cost(self):
        return self.monitor_cost_func( self )

    def get_extra_updates(self):
        return []

    def get_gparams(self):
        if not hasattr(self, 'gprams'):
            self.gparams = []
            grads = T.grad(self.get_cost(), self.get_params())
            for grad, para in zip(grads, self.get_params()):
                self.gparams.append((grad, para))
        return self.gparams

    def get_inputShape(self):
        return self.layers[0].get_inputShape()

    def get_outputShape(self):
        return self.layers[-1].get_outputShape()

    def feed_forward(self, input):
        if not hasattr(self, 'predict_fn'):
            self.predict_fn = theano.function(
                inputs=[self.get_inputTensor()], outputs=self.get_outputTensor()
            )
        return self.predict_fn(input)
