from dataPool import dataPool

class filterPool(dataPool):
    '''
    Pre-process the data before returning it
    '''
    def __init__(self, reader, filter):
        dataPool.__init__(self, reader)
        self.filter = filter

    def read_instances(self, batchSize=1):
        '''
        Always returns a list
        '''
        dataList = dataPool.read_instances(self, batchSize)
        filterList = []
        for data in dataList:
            filterList.append(self.filter.filter(data))
        return filterList
