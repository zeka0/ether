import numpy as np
from ether.instance.instance import instance

'''
If you don't want to filter the instances
Simply use other pools
'''

class filterBase:
    def filter(self, data):
        raise NotImplementedError()

class normFilter(filterBase):
    '''
    Normalize the values
    '''
    def __init__(self, **kwargs):
        assert kwargs.has_key('min')
        assert kwargs.has_key('max')
        assert kwargs.has_key('should_convert')
        self.min = kwargs['min']
        self.max = kwargs['max']
        self.sdConvert = kwargs['should_convert']

    def filter(self, data):
        if isinstance(data, instance):
            if self.sdConvert:
                data.reset_attr( np.array(data.get_attr(), dtype=np.double) )
            datax = data.get_attr()
            datax = (datax - self.min) / (self.max - self.min)
            data.reset_attr(datax)
            return data
        else:
            if self.sdConvert:
                data = np.array(data, dtype=np.double)
            return (data - self.min) / (self.max - self.min)

class picFilter(filterBase):
    '''
    Change pixies to certain values
    '''
    def __init__(self, **kwargs):
        assert kwargs.has_key('grey_num')
        assert kwargs.has_key('white_num')
        self.grey_num = kwargs['grey_num']
        self.white_num = kwargs['white_num']

    def filter(self, data):
        '''
        grey_num is to specify the number to substitute the grey pixies in the image.
        white_num is to specify the number to substitute the white pixies in the image.
        '''
        datax = data.get_attr()
        boolArr = (datax == 0)
        datax[boolArr] = self.white_num
        boolArr = (boolArr == False) #reverse the boolArr
        datax[boolArr] = self.grey_num
        data.reset_attr(datax)
        return data

class dimFilter(filterBase):
    def __init__(self, neo_shape):
        self.neo_shape = neo_shape

    def filter(self, data):
        data.reset_attr(data.get_attr().reshape(self.neo_shape))
        return data
