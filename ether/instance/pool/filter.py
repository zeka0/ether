import numpy as np
from ether.instance.instance import instance
import nltk

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

class nltkFilter(filterBase):
    '''
    @Deprecated
    Because nltkFilter has no way to bypass splitting sentences, it's deprecated
    '''
    sentence_start_token = 'SENTENCE_START'
    sentence_end_token = 'SENTENCE_END'
    unknown_token = 'UNKOEN_TOKEN'
    vocabulary_size = 8000

    '''
    Used to convert strings of letters to numbers
    With start-token and end-token
    '''
    def __init__(self):
        pass

    def filter(self, data):
        datax = data.get_attr()
        datax = '%s %s %s' % (nltkFilter.sentence_start_token, datax, nltkFilter.sentence_end_token)
        datax = nltk.sent_tokenize(datax)
        word_freq = nltk.FreqDist(datax)
        vocab = word_freq.most_common(nltkFilter.vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(nltkFilter.unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        datax = [w if w in word_to_index else nltkFilter.unknown_token for w in datax]

        '''The target of a sentence is one space shift of the original sentence'''
        train = np.asarray([[word_to_index[w] for w in datax[:-1]]])
        target = np.asarray([[word_to_index[w] for w in datax[1:]]])
        data.reset_attr(train)
        data.reset_tar(target)

