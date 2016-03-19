from nnet.util.exception import *
class trainer:
    def __init__(self, dataBase, optimizer, validator, nnet):
        self.optimizer=optimizer
        self.optimizer.set_owner(nnet)
        self.validator=validator
        self.validator.set_owner(nnet)
        self.dataBase=dataBase #Storage

    def set_validator(self, validator):
        self.validator = validator
        self.validator.set_owner(self)

    def read_instances(self, batchSize=1):
        '''
        Return target vec
        '''
        if self.dataBase.has_nextInstance(batchSize):
            return self.dataBase.read_instances(batchSize)
        else: raise instanceException('nnet has no more instances')

    def has_nextInstance(self, batchSize=1):
        return self.dataBase.has_nextInstance(batchSize)

    def train(self, cycles):
        for i in xrange(cycles):
            if self.has_nextInstance(1):
                instanceList = self.read_instances(1)
                self.optimizer.train_once(instanceList[0].get_attr(), instanceList[0].get_target())

    def validate(self, cycles):
        for i in xrange(cycles):
            if self.has_nextInstance(1):
                instanceList = self.read_instances(1)
                self.validator.validate_once(instanceList[0].get_attr(), instanceList[0].get_target())
        return self.validator.get_error_rate()
