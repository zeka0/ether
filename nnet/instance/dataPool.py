from collections import deque

class dataPool(object):
    def __init__(self, reader):
        self.instance_queue = deque()
        self.set_reader(reader)

    def set_reader(self, reader):
        self.reader = reader

    def get_reader(self):
        return self.reader

    def __pop_queue__(self, batchSize=1):
        for i in xrange(batchSize):
            yield self.instance_queue.pop()

    def __append_queue__(self, instances):
        if isinstance(instances, list):
            self.instance_queue.extend(instances)
        else: self.instance_queue.append(instances)

    def __extend_queue__(self, gen):
        self.instance_queue.extend(gen)

    def get_size(self):
        return len(self.instance_queue)

    def has_nextInstance_now(self, batchSize=1):
        '''
        Return whether the pool has the instances now
        That means without requiring the reader to read
        '''
        return len(self.instance_queue) != 0

    def read_instances(self, batchSize=1):
        '''
        Always returns a list
        '''
        tmpHold=[ins for ins in self.__pop_queue__(batchSize)]
        return tmpHold

    def has_nextInstance(self, batchSize):
        hasNext=self.has_nextInstance_now(batchSize) #For stochastic pool, refill
        if hasNext: return True
        hasNext = self.reader.has_nextInstance(batchSize - self.get_size())
        if hasNext:
            self.__extend_queue__( self.reader.read_instance(batchSize - self.get_size()) )
        return hasNext

    def drain_reader(self):
        '''
        Read all instances in the reader
        '''
        if self.reader.has_nextInstance(1):
            self.__append_queue__( self.reader.read_all() )
