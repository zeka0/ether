class dataReader(object):
    '''
    Interface+Buffering of the instances
    SubClass should implement the "read_source(self, size)" method
    '''
    def get_numOf_Attrs(self):
        '''
        Return the number of attributes
        '''
        raise NotImplementedError()

    def get_numOf_Targets(self):
        '''
        Return the number of targets
        '''
        raise NotImplementedError()

    def read_instance(self, batchSize):
        '''
        It's required to return a list
        And returning a generator is also encouraged
        '''
        raise NotImplementedError()

    def has_nextInstance(self, size):
        '''
        SubClass should implements this to tell if the reader can read extra 'size' instances
        '''
        raise NotImplementedError()

    def read_all(self):
        '''
        Read all the instances availble
        '''
        raise NotImplementedError()