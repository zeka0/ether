from theano.tensor.signal.conv import conv2d
from theano import tensor as T
import theano
from nnet.util.shape import *

from core import *
from nnet.mlp.initialize import *

'''
Personally I recommend using T.signal.conv2d instead of using T.nnet.conv2d
Because it's much more simple, and more flexible than nnet.conv2d
And it can be used to implement LeNet-5 with greater ease

As for conv2DLayer and mergeLayer
You can find that this two layers don't support the set_inputTensor and get_inputTensor
But their conterparts are provided
Since the layer-connection only requires the outputTensor
These changes aren't harmful
'''

class conv2DLayer(layer):
    '''
    When viewing this conv2DLayer, you can think it's output as a single feature-map
    And in order to make several feature-maps out of a single 2D image
    Just create more conv2DLayer and make them connect to the image
    '''
    def __init__(self, filterShape):
        layer.__init__(self)
        if len( filterShape ) != 2:
            raise mlpException('wrong shape parameter in convolution layer')
        self.filterShape = filterShape

    def init_bias(self):
        self.bias = shared.shared_ones( self.get_outputShape() )

    def get_filterShape(self):
        return self.filterShape

    def get_numOfFilters(self):
        return len( self.get_preLayers() )

    def init_filters(self):
        self.filters = normal.init_filters( self.get_numOfFilters(), self.get_filterShape() )

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

    def connect(self, *layers):
        self.set_preLayers(layers)
        self.init_filters()
        self.outputShape = conv2D_shape( layers[0].get_outputShape(), self.get_filterShape() )
        self.init_bias()

        outputTensor = self.get_bias()
        inputTensors = []
        #one filter for every pre-layer
        for i, filter in zip( xrange( self.get_numOfFilters() ), self.get_filters() ):
            outputTensor += conv2d( layers[i].get_outputTensor(), filters=filter )
            inputTensors.append( layers[i].get_outputTensor() )

        self.set_outputTensor( outputTensor )
        self.set_inputTensors( inputTensors )

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

class subSampleLayer(layer):
    def __init__(self, subSampleShape):
        layer.__init__(self)
        assert len(subSampleShape) == 2
        self.subSampleShape = subSampleShape

    def get_params(self):
        return None

    def verify_shape(self):
        pass

    def connect(self, *layers):
        assert len(layers) == 1
        self.intputShape = layers[0].get_outputShape()

        self.set_inputTensor( layers[0].get_outputTensor() )
        filter = np.ones( (1, 1) )
        outputTensor = conv2d( self.get_inputTensor(), filters=filter, subsample=self.subSampleShape )
        self.set_outputTensor( outputTensor )

    def get_inputShape(self):
        return self.intputShape

    def get_outputShape(self):
        return subSample_shape( self.get_inputShape(), self.subSampleShape )