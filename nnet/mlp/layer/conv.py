import numpy as np
from nnet.mlp.initialize import init_shared
from nnet.mlp.layer.core import *
from nnet.util.shape import *
from theano.tensor.signal.downsample import max_pool_2d

class conv2DLayer(layer):
    '''
    When viewing this conv2DLayer, you can think it's output as a single feature-map
    And in order to make several feature-maps out of a single 2D image
    Just create more conv2DLayer and make them connect to the image
    '''
    def __init__(self, n_fm, filterShape, border_mode, **kwargs):
        '''
        the bias can't be a scala
        '''
        layer.__init__(self)
        assert len(filterShape) == 2
        self.filterDs = filterShape
        self.n_fm = n_fm
        self.border_mode = border_mode

        assert kwargs.has_key('bias')
        assert kwargs.has_key('filter')
        self.biasKwargs = kwargs['bias']
        self.filterKwargs = kwargs['filter']
        assert not self.filterKwargs.has_key('shape')
        assert not self.biasKwargs.has_key('shape')
        assert not self.biasKwargs['distr'] == 'scala'

    def init_bias(self):
        '''
        :var n_fm means the number of this layer's feature maps
        '''
        self.bias = init_shared(shape=(self.n_fm,), **self.biasKwargs)

    def init_filters(self, n_pre_fm):
        self.filterShape = (self.n_fm, n_pre_fm) + self.filterDs
        self.filters = init_shared(shape=self.get_filterShape(), **self.filterKwargs)

    def get_filterShape(self):
        return self.filterShape

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.outputShape

    def get_params(self):
        paramList = []
        paramList.append( self.filters )
        paramList.append( self.bias )
        return paramList

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        assert len(self.inputShape) == 4
        self.set_inputTensor(layers[0].get_outputTensor())

        #compute outputTensor
        n_pre_fm = self.inputShape[1]
        self.init_filters(n_pre_fm)
        self.init_bias()
        outputTensor = self.bias.dimshuffle('x', 0, 'x', 'x') + T.nnet.conv2d(input=self.get_inputTensor(), filters=self.filters, border_mode=self.border_mode)
        self.set_outputTensor( outputTensor )

        #caculate shape
        out_img_shape = conv2D_shape((self.inputShape[2], self.inputShape[3]), self.filterDs, self.border_mode)
        self.outputShape = (self.inputShape[0], self.n_fm) + out_img_shape

class maxPoolLayer(layer):
    def __init__(self, poolShape, ignore_border):
        layer.__init__(self)
        self.ignore_border = ignore_border
        assert len(poolShape) == 2
        self.poolShape = poolShape

    def get_params(self):
        return None

    def get_inputShape(self):
        return self.inputShape

    def get_outputShape(self):
        return self.outputShape

    def connect(self, *layers):
        assert len(layers) == 1
        self.inputShape = layers[0].get_outputShape()
        assert len(self.inputShape) >= 2

        self.set_inputTensor( layers[0].get_outputTensor() )
        outputTensor = max_pool_2d(self.get_inputTensor(), ds=self.poolShape, ignore_border=self.ignore_border)
        self.set_outputTensor( outputTensor )
        pool_shape = maxPool_shape( ( self.inputShape[-2], self.inputShape[-1]) , self.poolShape, self.ignore_border )
        self.outputShape = self.inputShape[0:-2] + pool_shape
