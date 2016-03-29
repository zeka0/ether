from nnet.util.util import nnetController
class tracker(nnetController):
    def __init__(self, opt):
        nnetController.__init__(self)
        self.optimizer = opt
        self.trackDic = dict()

    def set_owner(self, nnet):
        nnetController.set_owner(self, nnet)
        self.optimizer.set_owner(nnet)
        self.init_track()
        for key in self.get_trackKeys():
            self.trackDic[key] = []

    def get_trackKeys(self):
        raise NotImplementedError()

    def init_track(self):
        '''
        Subclasses override this to provide initializations
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
        assert not isinstance(key, list)
        assert not isinstance(value, list)
        self.trackDic[value].append(key)

    def track_list(self, keys, values):
        assert isinstance(values, list)
        assert len(keys) == len(values)
        for key, value in zip(keys, values):
            self.track_tuple((key, value))

    def train_once(self, attr, tar):
        '''
        Wrapper over optimizer
        '''
        raise NotImplementedError()

