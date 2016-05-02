from ether.util.exception import *
class trainer:
    def __init__(self, dataBase, optimizer, validator, model):
        self.model = model
        self.set_optimizer(optimizer)
        self.set_validator(validator)
        self.set_dataBase(dataBase)

    def compile(self):
        if self.optimizer is not None:
            self.optimizer.set_owner(self.model)
        if self.validator is not None:
            self.validator.set_owner(self.model)

    def set_dataBase(self, dataBase):
        self.dataBase=dataBase #Storage

    def set_optimizer(self, optimizer):
        self.optimizer=optimizer

    def set_validator(self, validator):
        self.validator = validator

    def get_optimizer(self):
        return self.optimizer

    def get_validator(self):
        return self.validator

    def get_model(self):
        return self.model

    def has_nextInstance(self, batchSize=1):
        return self.dataBase.has_nextInstance(batchSize)

    def read_instances(self, batchSize=1):
        '''
        Return target vec
        '''
        if self.dataBase.has_nextInstance(batchSize):
            return self.dataBase.read_instances(batchSize)
        else: raise instanceException('model has no more instances')

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
