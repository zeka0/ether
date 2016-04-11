import numpy as np
from nnet.instance.instance import instance
class filterBase:
    def filter(self, data):
        raise NotImplementedError()

class normFilter(filterBase):
    '''
    Normalize the values
    '''
    def __init__(self, min, max, shouldConvert):
        self.min = min
        self.max = max
        self.sdConvert = shouldConvert

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
    def filter(self, data, grey_num, white_num):
        '''
        grey_num is to specify the number to substitute the grey pixies in the image.
        white_num is to specify the number to substitute the white pixies in the image.
        '''
        datax = data.get_attr()
        boolArr = (datax == 0)
        datax[boolArr] = white_num
        boolArr = (boolArr == False) #reverse the boolArr
        datax[boolArr] = grey_num
        data.reset_attr(datax)
        return data
