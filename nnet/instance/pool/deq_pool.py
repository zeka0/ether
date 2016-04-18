from collections import deque
from core import poolBase

class deqPool(poolBase):
    def __init__(self, reader):
        poolBase.__init__(self)
        self.instance_queue = deque()
        self.set_reader(reader)

    def set_reader(self, reader):
        self.reader = reader

    def get_reader(self):
        return self.reader

    def pop_queue(self, batchSize=1):
        for i in xrange(batchSize):
            yield self.instance_queue.pop()

    def append_queue(self, instances):
        if isinstance(instances, list):
            self.instance_queue.extend(instances)
        else: self.instance_queue.append(instances)

    def extend_queue(self, gen):
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
        tmpHold=[ins for ins in self.pop_queue(batchSize)]
        return tmpHold

    def has_nextInstance(self, batchSize):
        hasNext=self.has_nextInstance_now(batchSize) #For stochastic pool, refill
        if hasNext: return True
        hasNext = self.reader.has_nextInstance(batchSize - self.get_size())
        if hasNext:
            self.extend_queue( self.reader.read_instance(batchSize - self.get_size()) )
        return hasNext

    def drain_reader(self):
        '''
        Read all instances in the reader
        '''
        if self.reader.has_nextInstance(1):
            self.append_queue( self.reader.read_all() )
