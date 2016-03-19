from fullPool import *
class stochasticPool(fullPool):
    '''
    Stochastic drawing instances
    When using stochastic pool for testing(validitying)
    You should treat the output of this pool simply as testing instances
    '''
    def __init__(self, instances):
        fullPool.__init__(self, instances, 0, True) #disnable sampling

    def set_instances(self, instances):
        self.train_instances.extend(instances)

    def is_stochastic(self):
        return True

    def set_state(self, isTrain):
        '''
        Ignore input
        Since stochastic pool doesn't hold complete data-set, there is no need to use this method
        '''
        self.isTrain = True

    def get_totalSize(self):
        raise NotImplementedError()
