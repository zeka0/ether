from core import tracker
class paramTracker(tracker):
    '''
    The tensor object are hashable
    That's why we are using it as the key
    '''
    def __init__(self, opt):
        tracker.__init__(self, opt)
        self.params = []

    def get_trackKeys(self):
        return self.params

    def init_track(self):
        pass

    def add_params(self, *params):
        self.params.extend(params)
        '''
        I used to use the shared_variable's method get_value()
        to speed up. But I found that this action will limit those parameters
        to be tracked. This version is a little bit slow but it's more generous
        Most importantly, it meets the requirement of the mergeTracker
        '''
        self.optimizer.add_additionalOutputs(*params)
