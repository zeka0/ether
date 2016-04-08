from core import tracker
'''
Provide utilities for tracking multiple targets
'''
class mergeTracker(tracker):
    def __init__(self, opt):
        tracker.__init__(self, opt)
        self.trackKeys = []
        self.trackers = []

    def add_trackers(self, *trackers):
        '''
        trackers must be pre-added-keys
        '''
        for tcker in trackers:
            assert not isinstance(tcker, mergeTracker)
        self.trackers.extend(trackers)

    def init_track(self):
        for tcker in self.trackers:
            self.trackKeys.extend(tcker.get_trackKeys())

    def get_trackKeys(self):
        return self.trackKeys
