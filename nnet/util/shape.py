from exception import *
'''
This modula is used by the layer package to determine the shape of trainable parameters
'''

def conv2D_shape(imageShape, filterShape, mode='valid'):
    if len(imageShape) != 2:
        raise shapeError('image shape must be of 2 dim')
    if len(filterShape) != 2:
        raise shapeError('image shape must be of 2 dim')
    return ( imageShape[0] - filterShape[0] + 1, imageShape[1] - filterShape[1] + 1 )

def weightMatrix_shape(inputShape, connectionNum):
    if isinstance( inputShape, tuple ) or isinstance( inputShape, list ):
        assert len(inputShape) == 2
        assert inputShape[0] == 1
        return (inputShape[1], connectionNum)
    else:
        return (inputShape, connectionNum)

def subSample_shape(imageShape, subSampleShape):
    assert len(imageShape) ==2
    assert len(subSampleShape) ==2
    assert imageShape[0] % subSampleShape[0] ==0
    assert imageShape[1] % subSampleShape[1] ==0
    return ( imageShape[0]/subSampleShape[0], imageShape[1]/subSampleShape[1] )

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
