class paramTracker:
    '''
    The tensor object are hashable
    That's why we are using it as the key
    '''
    def __init__(self):
        self.trackDic = dict()

    def track(self, params):
        for para in params:
            if self.trackDic.get(para) is None:
                self.trackDic[para] = []
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
