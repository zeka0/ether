from ether.component.exception import *
from ether.instance.instance import *

mini_batch_size = 1

class trainer:
    def __init__(self, dataBase, optimizer, validator, model,
                 train_ep=None, valid_ep=None):
        '''
        :param train_ep: Extra-Operation during traning process, called once per training
        :type train_ep: EpBase
        :param valid_ep: Extra-Operation during validating process, called once per validate
        :type valid_ep: EpBase
        :param mini_batch_size: the size of mini-batch
        '''
        self.model = model
        self.set_optimizer(optimizer)
        self.set_validator(validator)
        self.set_dataBase(dataBase)
        self.train_ep = train_ep
        self.valid_ep = valid_ep
        self.mini_batch_size = mini_batch_size

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
        try:
            for i in xrange(cycles):
                if self.train_ep is not None:
                    self.train_ep.call()#call train_ep

                if self.has_nextInstance(self.mini_batch_size):
                    instanceList = self.read_instances(self.mini_batch_size)
                    ins = make_batch(instanceList)
                    self.optimizer.train_once(ins.get_attr(), ins.get_target())

        except instanceException:
            #no more instances available
            print 'exception occured in instance availability'

    def validate(self, cycles):
        try:
            for i in xrange(cycles):
                if self.valid_ep is not None:
                    self.valid_ep.call()#call valid_ep
                if self.has_nextInstance(1):
                    instanceList = self.read_instances(1)
                    self.validator.validate_once(instanceList[0].get_attr(), instanceList[0].get_target())
            return self.validator.get_error_rate()
        except instanceException:
            #no more instances available
            print 'exception occured in instance availability'
