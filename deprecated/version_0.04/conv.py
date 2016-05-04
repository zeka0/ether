import numpy as np
from theano.tensor.signal.conv import conv2d
from ether.component.init import init_shared
from ether.component.layer.core import *
from ether.util.shape import *

class conv2DLayer(layer):
    '''
    When viewing this conv2DLayer, you can think it's output as a single feature-map
    And in order to make several feature-maps out of a single 2D image
    Just create more conv2DLayer and make them connect to the image
    '''
    def __init__(self, filterShape, **kwargs):
        layer.__init__(self)
        if len( filterShape ) != 2:
            raise mlpException('wrong shape parameter in convolution layer')
        self.filterShape = filterShape
        assert kwargs.has_key('bias')
        assert kwargs.has_key('filter')
        self.biasKwargs = kwargs['bias']
        self.filterKwargs = kwargs['filter']
        assert not self.filterKwargs.has_key('shape')

    def init_bias(self):
        self.bias = init_shared(**self.biasKwargs)

    def init_filters(self):
        self.filters = []
        for i in xrange(self.get_numOfFilters()):
            self.filters.append( init_shared(shape=self.get_filterShape(), **self.filterKwargs) )

    def get_filterShape(self):
        return self.filterShape

    def get_numOfFilters(self):
        return len( self.get_preLayers() )

    def get_filters(self):
        return self.filters

    def get_bias(self):
        return self.bias

    def get_preLayers(self):
        return self.preLayers

    def set_preLayers(self, layers):
        self.preLayers = layers

    def set_inputTensor(self, inputTensor):
        raise NotImplementedError('conv2DLayer has multiple inputTensors')

    def get_inputTensor(self):
        raise NotImplementedError('conv2DLayer has multiple inputTensors')

    def set_inputTensors(self, inputTensors):
        self.inputTensors = inputTensors

    def get_inputTensors(self):
        return self.inputTensors

    def get_inputShape(self):
        return ( len(self.get_preLayers()), self.get_preLayers()[0].get_outputShape[0], self.get_preLayers()[0].get_outputShape[1] )

    def get_outputShape(self):
        return self.outputShape

    def verify_shape(self):
        baseShape = self.get_preLayers()[0].get_outputShape()
        for layer in self.get_preLayers():
            if layer.get_outputShape() != baseShape:
                raise shapeError(self, 'conv2D pre-layers\'shape must be the same')

    def get_params(self):
        paramList = []
        paramList.extend( self.get_filters() )
        paramList.append( self.get_bias() )
        return paramList

    def connect(self, *layers):
        self.set_preLayers(layers)
        self.init_filters()
        self.outputShape = conv2D_shape( layers[0].get_outputShape(), self.get_filterShape() )
        self.init_bias()

        outputTensor = self.get_bias()
        inputTensors = []
        #one filter for every pre-layer
        for i, filter in zip( xrange( self.get_numOfFilters() ), self.get_filters() ):
            outputTensor = outputTensor + conv2d( layers[i].get_outputTensor(), filters=filter )
            inputTensors.append( layers[i].get_outputTensor() )

        self.set_outputTensor( outputTensor )
        self.set_inputTensors( inputTensors )

class subSampleLayer(layer):
    def __init__(self, subSampleShape, **kwargs):
        layer.__init__(self)
        assert len(subSampleShape) == 2
        self.subSampleShape = subSampleShape
        assert kwargs.has_key('coef')
        assert kwargs.has_key('bias')
        self.coef = init_shared(**kwargs['coef'])
        self.bias = init_shared(**kwargs['bias'])

    def get_params(self):
        return [self.bias, self.coef]

    def verify_shape(self):
        pass

    def get_inputShape(self):
        return self.intputShape

    def get_outputShape(self):
        return subSample_shape( self.get_inputShape(), self.subSampleShape )

    def connect(self, *layers):
        assert len(layers) == 1
        self.intputShape = layers[0].get_outputShape()

        self.set_inputTensor( layers[0].get_outputTensor() )
        filter = np.ones( (1, 1) )
        outputTensor = self.coef * conv2d( self.get_inputTensor(), filters=filter, subsample=self.subSampleShape ) + self.bias
        self.set_outputTensor( outputTensor )
