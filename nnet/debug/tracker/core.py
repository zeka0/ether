from nnet.util.util import nnetController
class tracker(nnetController):
    def __init__(self, opt):
        nnetController.__init__(self)
        self.optimizer = opt
        self.trackDic = dict()

    def set_owner(self, nnet):
        nnetController.set_owner(self, nnet)
        '''
        Be cautious that the init_track is called before the optimizer is actually set to nnet
        That means all the methods in base nnetController is invalid
        '''
        self.init_track()
        self.optimizer.set_owner(nnet)
        for key in self.get_trackKeys():
            self.trackDic[key] = []

    def get_trackDict(self):
        return self.trackDic

    def get_trackKeys(self):
        raise NotImplementedError()

    def init_track(self):
        '''
        Subclasses override this to provide initializations
        It must be followed that in this method
        The subclass should only add additional parameters to the optmizer
        And can't call any of the methods of the nnetController class
        '''
        raise NotImplementedError()

    def track(self, key, value):
        if isinstance(key, list):
            self.track_list(key, value)
        elif isinstance(key, tuple):
            self.track_tuple(key, value)
        else: self.trackDic[key].append(value)

    def track_tuple(self, keyValueTuple):
        key = keyValueTuple[0]
        value = keyValueTuple[1]
        self.trackDic[key].append(value)

    def track_list(self, keys, values):
        assert len(keys) == len(values)
        for key, value in zip(keys, values):
            self.track_tuple((key, value))

    def train_once(self, attr, tar):
        '''
        Wrapper over optimizer
        '''
        result = self.optimizer.train_once(attr, tar)
        for key, value in zip(self.get_trackKeys(), result):
            self.track(key, value)

    def print_info(self, maxCycles=3, comp=False):
        for para in self.get_trackKeys():
            print 'Key name:', para
            print 'Train traits:'
            if comp:
                if len(self.trackDic[para]) == 0:
                    print 'Skip key', para
                else:
                    base = self.trackDic[para][0]
                    print 'Base_value', base
                    for i in xrange(1, maxCycles):
                        if i < len(self.trackDic[para]):
                            print 'Cycle ', i, 'Comp result:'
                            print base == self.trackDic[para][i]
            else:
                for i in xrange( maxCycles):
                    if i < len(self.trackDic[para]):
                        print 'Cycle ', i, 'Values:'
                        print self.trackDic[para][i]

