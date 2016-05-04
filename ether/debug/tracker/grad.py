import theano
from core import tracker

'''
It's not suggested to use this module
Because it can slow down the process quite a lot
'''

class gradTracker(tracker):
    '''
    The tensor object are hashable
    That's why we are using it as the key
    '''
    def __init__(self, opt):
        '''
        Plays a trick in intialization
        '''
        tracker.__init__(self, opt)
        self.paras = []

    def add_gradParaTuple(self, *gradParaTuples):
        grads = []
        for grad, para in gradParaTuples:
            self.paras.append(para)
            grads.append(grad)
        self.optimizer.add_additionalOutputs(*grads)

    def get_trackKeys(self):
        return self.paras

    def init_track(self):
        pass
