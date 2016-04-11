##Introduction
The layer is the key components of the nnet. It builds the tensors required for the computation of the theano.

##Forward notes
###kwargs for layer specific initialization
  In some layers, it's required that you pass the __kwargs__. It's used to initialize some specific parameters for the layer. Like in conv2d, you need to decide the shape of the filters, whether it's of normal distribution or universal distribution and so on. However, even that is not enough.As several layers has more than 1 parameters, the kwargs are required to be a dict of dict.
  Example:
```python
weightLayer( bias={'func':'scala', 'value':0, 'type':float}, weight={'func':'normal', 'low':-0.25, 'high':0.25})
```

###Initialiation of the parameters
- __distr__
  - __normal__ distribution required parameters
    - __shape__
    - __mean__
    - __std__
  - __uniform__ distribution required parameters
    - __shape__
    - __low__
    - __high__
  - __scala__ required parameters
    - __type__
    - __value__
  - __constant__ distribution required parameters
    - __shape__
    - __type__
    - __value__

##Modulas & Classes
###class layer
- __method connect__ 
  It is key in the layers. It will build the connection of two layers, compute the tensor, initialize parameters and so on.

###class activationLayer

###class conv2DLayer
Personally I recommend using T.signal.conv2d instead of using T.nnet.conv2d, because it's much more simple, and more flexible than nnet.conv2d. And it can be used to implement LeNet-5 with greater ease. As for conv2DLayer and mergeLayer, you can find that this two layers don't support the set_inputTensor and get_inputTensor. But their conterparts are provided. Since the layer-connection only requires the outputTensor. These changes aren't harmful.

###class merge1DLayer
__merge1DLayer__ will concatenate several outputs of previous layers. It can only support 1D ( which we mean tensors of shape(1, x) ).


###class weightLayer
