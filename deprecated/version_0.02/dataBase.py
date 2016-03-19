from nnet.instance.pool import *

class dataBase:
    '''
    After the construction of dataBase
    You should neither call stochastic_init or full_init to finish construction
    You should always call "has_nextInstance" before call "get_nextInstance"
    Calling them with same-parameters
    Otherwise be-prepared to catch exceptions
    '''
    def __init__(self, reader):
        self.reader = reader

    def set_state(self, isTrain):
        self.pool.set_state(isTrain)

    def read_instances(self, batchSize):
        '''
        Filter the instances if needed
        '''
        instances=[]
        for ins in self.reader.read_instance(batchSize):
            instances.append(ins)
        return instances

    def stochastic_init(self, initInstanceNum=100):
        self.pool = stochasticPool(self.__read_instances(initInstanceNum))

    def full_init(self, sampleRate=0.1, initInstanceNum=1000, isTrain=True):
        self.pool = fullPool(self.__read_instances(initInstanceNum), sampleRate, isTrain)

    def get_numOf_Attrs(self):
        return self.reader.get_numOf_Attrs()

    def get_numOf_Targets(self):
        return self.reader.get_numOf_Targets()

    def has_nextInstance(self, batchSize):
        if not self.isStochastic:
            return self.pool.has_nextInstance(batchSize)
        hasNext=self.pool.has_nextInstance(batchSize) #For stochastic pool, refill
        if hasNext: return True
        instances = self.__read_instances(self.pool.get_max_bufferSize()-self.pool.get_bufferSize())
        self.pool.set_instances(instances) #For stochastic pool
        return self.pool.has_nextInstance(batchSize)

    def get_nextInstance(self, batchSize=1):
        '''
        May throw instanceException
        '''
        return self.pool.get_nextInstance(batchSize)

    def get_bufferSize(self):
        return self.pool.get_bufferSize()

    def get_attrTar_tuple(self):
        return (self.reader.get_attrName(), self.reader.get_tarName())
