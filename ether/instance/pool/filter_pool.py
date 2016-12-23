from core import poolBase

class filterPool(poolBase):
    '''
    Pre-process the ins before returning it.
    Since the subclass of poolBase can be so many, I used a different mechanism here.
    The pool arguement must a subclass of poolBase.

    Now support a list of filters, applied from index 0 to index -1 iterately
    '''
    def __init__(self, pool, flist):
        assert isinstance(pool, poolBase)
        self.pool = pool
        self.flist = flist

    def has_nextInstance(self, batchSize):
        return self.pool.has_nextInstance(batchSize)

    def has_nextInstance_now(self, batchSize=1):
        return self.pool.has_nextInstance_now(batchSize)

    def read_instances(self, batchSize=1):
        '''
        Always returns a list
        '''
        insList = self.pool.read_instances(batchSize)
        filterList = []
        for ins in insList:
            for filter in self.flist:
                ins = filter.filter(ins)
            filterList.append(ins)
        return filterList
