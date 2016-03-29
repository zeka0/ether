from core import tracker
class paramTracker(tracker):
    '''
    The tensor object are hashable
    That's why we are using it as the key
    '''
    def __init__(self, opt):
        tracker.__init__(self, opt)

    def get_trackKeys(self):
        return self.get_params()

    def init_track(self):
        pass

    def train_once(self, attr, tar):
        self.optimizer.train_once(attr, tar)
        for para in self.trackDic.keys():
            #we don't use extend here because kwargs[key] may be an ndarray which can be a list
            self.trackDic[para].append(para.get_value())

    def print_info(self, maxCycle=3):
        for para in self.trackDic:
            print 'parameter name:\n', para
            print 'train traits:'
            for i in xrange( maxCycle):
                if i < len(self.trackDic[para]):
                    print 'Cycle ', i, ' Value:\n'
                    print self.trackDic[para][i]
