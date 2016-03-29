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

    def set_tuples(self, gradParaTuples):
        grads = []
        self.paras = []
        for grad, para in gradParaTuples:
            self.paras.append(para)
            grads.append(grad)
        self.optimizer.add_additionalOutputs(grads)

    def get_trackKeys(self):
        return self.paras

    def init_track(self):
        pass

    def set_owner(self, nnet):
        self.set_owner(nnet)
        self.optimizer.set_owner(nnet)

    def train_once(self, attr, tar):
        result = self.optimizer.train_once(attr, tar)
        for para, grad in zip(self.get_trackKeys(), result):
            self.track(para, grad)

    def print_info(self, maxCycle=3, comp=False):
        for para in self.trackDic:
            print 'parameter name:\n', para
            #print 'final value:\n', para.get_value()
            print 'train traits:'
            if comp:
                base = self.trackDic[para][0]
                print '1st_cycle gradients', base
                for i in xrange(1, maxCycle):
                    if i < len(self.trackDic[para]):
                        print 'cycle ', i, 'compare with cycle 1 gradients, True means grads are the same, False means grads are not the same'
                        print base == self.trackDic[para][i]
            else:
                for i in xrange( maxCycle):
                    if i < len(self.trackDic[para]):
                        print 'cycle ', i, 'Values:'
                        print self.trackDic[para][i]
