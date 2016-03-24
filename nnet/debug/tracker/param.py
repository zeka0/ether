class paramTracker:
    '''
    The tensor object are hashable
    That's why we are using it as the key
    '''
    def __init__(self):
        self.trackDic = dict()

    def set_track_params(self, params):
        for para in params:
            if self.trackDic.get(para) is None:
                self.trackDic[para] = []

    def track_once(self):
        for para in self.trackDic.keys():
            #we don't use extend here because kwargs[key] may be an ndarray which can be a list
            self.trackDic[para].append(para.get_value())

    def print_info(self, maxCycle=3):
        for para in self.trackDic:
            print 'parameter name:\n', para
            #print 'final value:\n', para.get_value()
            print 'train traits:'

            base = self.trackDic[para][0]
            for i in xrange( maxCycle):
                if i < len(self.trackDic[para]):
                    print 'Cycle ', i, ' Value:\n'
                    print self.trackDic[para][i]

class paramTrackOptimizer(paramTracker):
    def __init__(self, opt):
        paramTracker.__init__(self)
        self.optimizer = opt

    def set_owner(self, nnet):
        self.optimizer.set_owner(nnet)

    def train_once(self, attr, tar):
        self.optimizer.train_once(attr, tar)
        self.set_track_params(self.optimizer.get_params())
        self.track_once()