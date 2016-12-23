import numpy as np

def make_batch(inslist):
    attr_list = []
    tar_list = []
    for ins in inslist:
        attr_list.append(ins.get_attr()[0])
        tar_list.append(ins.get_target()[0])
    return instance(attr_list, tar_list)

class instance(object):
    '''
    instance is not responsible for shaping the vector
    '''
    def __init__(self, vecOfAttr, vecOfTarget):
        self.vecOfAttr=vecOfAttr
        self.vecOfTarget=vecOfTarget

    def get_attr(self):
        return self.vecOfAttr

    def get_target(self):
        return self.vecOfTarget

    def reset_attr(self, vecOfAttr):
        self.vecOfAttr = vecOfAttr

    def reset_tar(self, vecOfTarget):
        self.vecOfTarget = vecOfTarget

class imgInstance(instance):
    '''
    Transform the target into a vector of possibilities
    '''
    def __init__(self, img, tar, numOfTars):
        vecPoss = np.zeros(numOfTars)
        vecPoss[tar] = 1
        instance.__init__( self, img, vecPoss )

class unlabledInstance:
    def __init__(self, vecOfAttr):
        self.vecOfAttr = vecOfAttr

    def get_attr(self):
        return self.vecOfAttr

    def reset_attr(self, vecOfAttr):
        self.vecOfAttr = vecOfAttr
