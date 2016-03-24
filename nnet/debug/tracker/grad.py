import theano

'''
It's not suggested to use this module
Because it can slow down the process quite a lot
'''

class gradTracker:
    '''
    The tensor object are hashable
    That's why we are using it as the key
    '''
    def __init__(self):
        self.trackDic = dict()

    def track_dict(self, **kwargs):
        for key in kwargs:
            if self.trackDic.get(key) is None:
                self.trackDic[key] = []
            #we don't use extend here because kwargs[key] may be an ndarray which can be a list
            self.trackDic[key].append(kwargs[key])

    def track_tuple(self, gradParaTuple):
        grad = gradParaTuple[0]
        para = gradParaTuple[1]
        assert not isinstance(grad, list)
        assert not isinstance(para, list)
        if self.trackDic.get(para) is None:
            self.trackDic[para] = []
        self.trackDic[para].append(grad)

    def track(self, grads, params):
        if isinstance(grads, list):
            assert isinstance(params, list)
            assert len(grads) == len(params)
            for grad, para in zip(grads, params):
                self.track_tuple((grad, para))
        else:
            self.track_tuple((grads, params))

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


class autoTracker(gradTracker):
    def __init__(self, gradParaTuples, inputs=[]):
        gradTracker.__init__(self)
        self.gradParaTuples = gradParaTuples
        self.inputs = inputs
        self.init_gradFuns()

    def init_gradFuns(self):
        self.paraFunDict = dict()
        for tup in self.gradParaTuples:
            grad = tup[0]
            para = tup[1]
            self.paraFunDict[para] = theano.function(inputs=self.inputs, outputs=grad)

    def track_once(self, *inputs):
        for para in self.paraFunDict:
            self.track( self.paraFunDict[para](*inputs), para )

class gradTrackAutoOptimizer(autoTracker):
    def __init__(self, opt):
        '''
        Plays a trick in intialization
        '''
        self.optimizer = opt

    def set_owner(self, nnet):
        self.optimizer.set_owner(nnet)
        autoTracker.__init__(self, self.optimizer.get_gradients(),
                             inputs=[self.optimizer.get_inputTensor(), self.optimizer.get_targetTensor()])

    def train_once(self, attr, tar):
        self.optimizer.train_once(attr, tar)
        self.track_once(attr, tar)