from core import poolBase

class filterPool(poolBase):
    '''
    Pre-process the data before returning it.
    Since the subclass of poolBase can be so many, I used a different mechanism here.
    The pool arguement must a subclass of poolBase.
    '''
    def __init__(self, pool, filter):
        assert isinstance(pool, poolBase)
        self.pool = pool
        self.filter = filter

    def has_nextInstance(self, batchSize):
        return self.pool.has_nextInstance(batchSize)

    def has_nextInstance_now(self, batchSize=1):
        return self.pool.has_nextInstance_now(batchSize)

    def read_instances(self, batchSize=1):
        '''
        Always returns a list
        '''
        dataList = self.pool.read_instances(batchSize)
        filterList = []
        for data in dataList:
            filterList.append(self.filter.filter(data))
        return filterList
