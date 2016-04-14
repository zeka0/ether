class poolBase:
    def has_nextInstance(self, batchSize):
        raise NotImplementedError()

    def has_nextInstance_now(self, batchSize=1):
        raise NotImplementedError()

    def read_instances(self, batchSize=1):
        raise NotImplementedError()
