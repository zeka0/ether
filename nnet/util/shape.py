from exception import *
import numpy as np
'''
This modula is used by the layer package to determine the shape of trainable parameters
'''

def conv2D_shape(imageShape, filterShape, mode):
    if len(imageShape) != 2:
        raise shapeError('image shape must be of 2 dim')
    if len(filterShape) != 2:
        raise shapeError('image shape must be of 2 dim')
    assert mode in ['valid', 'full']
    if mode == 'valid':
        return ( imageShape[0] - filterShape[0] + 1, imageShape[1] - filterShape[1] + 1 )
    elif mode == 'full':
        return ( imageShape[0] + filterShape[0] - 1, imageShape[1] + filterShape[1] - 1 )

def weightMatrix_shape(inputShape, connectionNum):
    assert len(inputShape) == 2
    return (inputShape[0], connectionNum)

#TODO:deprecate
def subSample_shape(imageShape, subSampleShape):
    assert len(imageShape) ==2
    assert len(subSampleShape) ==2
    assert imageShape[0] % subSampleShape[0] ==0
    assert imageShape[1] % subSampleShape[1] ==0
    return ( imageShape[0]/subSampleShape[0], imageShape[1]/subSampleShape[1] )

def maxPool_shape(imageShape, poolShape, ignore_border):
    assert len(imageShape) ==2
    assert len(poolShape) ==2
    if ignore_border:
        return ( np.floor( float(imageShape[0])/poolShape[0] ), np.floor( float(imageShape[1])/imageShape[1] ) )
    else:
        return ( np.ceil( float(imageShape[0])/poolShape[0] ), np.ceil( float(imageShape[1])/imageShape[1] ) )

def flatten_shape(oriShape, outdim=1):
    assert isinstance(oriShape, tuple)
    assert outdim >= 1
    assert len(oriShape) >= outdim
    outShape = []
    for i in xrange(0, outdim - 1):
        outShape.append( oriShape[i] )
    lastDimSize = 1
    for i in xrange(outdim - 1, len(oriShape)):
        lastDimSize *= oriShape[i]
    outShape.append(lastDimSize)
    if len(outShape) > 1:
        return tuple(outShape)
    else: return (1, outShape[0])

def dimShuffle_shape(oriShape, *pattern):
    #TODO
    pass
