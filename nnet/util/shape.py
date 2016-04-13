from exception import *

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
