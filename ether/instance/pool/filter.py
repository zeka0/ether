import numpy as np
from ether.instance.instance import instance
import nltk

'''
If you don't want to filter the instances
Simply use other pools
'''

class filterBase:
    def filter(self, ins):
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

    def filter(self, ins):
        if isinstance(ins, instance):
            if self.sdConvert:
                ins.reset_attr( np.array(ins.get_attr(), dtype=np.double) )
            insx = ins.get_attr()
            insx = (insx - self.min) / (self.max - self.min)
            ins.reset_attr(insx)
            return ins
        else:
            if self.sdConvert:
                ins = np.array(ins, dtype=np.double)
            return (ins - self.min) / (self.max - self.min)

class picFilter(filterBase):
    '''
    Change pixies to certain values
    '''
    def __init__(self, **kwargs):
        assert kwargs.has_key('grey_num')
        assert kwargs.has_key('white_num')
        self.grey_num = kwargs['grey_num']
        self.white_num = kwargs['white_num']

    def filter(self, ins):
        '''
        grey_num is to specify the number to substitute the grey pixies in the image.
        white_num is to specify the number to substitute the white pixies in the image.
        '''
        insx = ins.get_attr()
        boolArr = (insx == 0)
        insx[boolArr] = self.white_num
        boolArr = (boolArr == False) #reverse the boolArr
        insx[boolArr] = self.grey_num
        ins.reset_attr(insx)
        return ins

class dimFilter(filterBase):
    '''
    Change the shape of instances
    If shape is None, it will maitain the original shape
    '''
    def __init__(self, neo_attr_shape=None, neo_tar_shape=None):
        self.neo_attr_shape = neo_attr_shape
        self.neo_tar_shape = neo_tar_shape

    def filter(self, ins):
        if self.neo_attr_shape is not None:
            ins.reset_attr(ins.get_attr().reshape(self.neo_attr_shape))
        if self.neo_tar_shape is not None:
            ins.reset_tar(ins.get_target().reshape(self.neo_tar_shape))
        return ins

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

    def filter(self, ins):
        insx = ins.get_attr()
        insx = '%s %s %s' % (nltkFilter.sentence_start_token, insx, nltkFilter.sentence_end_token)
        insx = nltk.sent_tokenize(insx)
        word_freq = nltk.FreqDist(insx)
        vocab = word_freq.most_common(nltkFilter.vocabulary_size-1)
        index_to_word = [x[0] for x in vocab]
        index_to_word.append(nltkFilter.unknown_token)
        word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

        insx = [w if w in word_to_index else nltkFilter.unknown_token for w in insx]

        '''The target of a sentence is one space shift of the original sentence'''
        train = np.asarray([[word_to_index[w] for w in insx[:-1]]])
        target = np.asarray([[word_to_index[w] for w in insx[1:]]])
        ins.reset_attr(train)
        ins.reset_tar(target)

