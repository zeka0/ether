from ether.instance.instance import instance
from ether.util import *
from core import dataReader
import _csv

class neoCsvReader(dataReader):
    def __init__(self, filePath):
        self.filePath = filePath

    def get_numOf_Attrs(self):
        raise NotImplementedError()

    def get_numOf_Targets(self):
        raise NotImplementedError()

    def read_instance(self, batchSize):
        raise NotImplementedError()

    def has_nextInstance(self, size):
        raise NotImplementedError()

    def read_all(self):
        with open(self.filePath, 'rb') as f:
            reader = _csv.reader(f, skipinitialspace=True)
            reader.next()
            tmp = [instance(x, None) for x in reader]
        return tmp
