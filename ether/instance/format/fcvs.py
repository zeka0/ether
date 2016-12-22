from ether.instance.instance import instance
from ether.util import *
from core import dataReader
import csv

def file_len(fpath):
    with open(fpath) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class csvReader(dataReader):
    '''
    hasTitle specifies whether the cvs file has title attributes
    attrSelected is a list specifying which column should be used
    tarSelected is a list specifing which column should be treated as targetValue
    '''
    def __init__(self, filePath,
                 maxQueueLen, attrSelected, tarSelected=-1,
                 headSpace=0, hasTitle=False):
        '''
        attrSelected should be a list
        tarSelected should specify a column number
        tarSelected == -1 means no targets provided
        maxLen is the maxium of the readable instance
        '''
        self.dataSource=open(filePath, 'rU') #Read Universial
        self.reader = csv.reader(self.dataSource, skipinitialspace=True)
        self.filePath = filePath
        self.attrSelected = attrSelected
        self.tarSelected = tarSelected
        self.hasTitle = hasTitle
        self.insNum = file_len(filePath) - headSpace
        self.insReadNum = 0
        '''
        Parsing and recording relevant infomation
        '''
        if hasTitle:
            attrNameVec, tarNameVec = self.get_colTar(self.reader.next())
            self.attrNameVec = attrNameVec
            self.targetNameVec = tarNameVec
            self.insNum -= 1
        for i in xrange(0, headSpace): #Discard lines
            self.reader.next()

    def get_numOf_Attrs(self):
        return len(self.attrSelected)

    def get_numOf_Targets(self):
        raise NotImplementedError()

    def read_instance(self, batchSize):
        tmp = []
        for i in xrange(batchSize):
            x = self.reader.next()
            tmp.append(self.build_instance(x))
        return tmp

    def has_nextInstance(self, size):
        return self.insReadNum + size <= self.insNum

    def read_all(self):
        tmp = [self.build_instance(x) for x in self.reader]
        self.insReadNum = self.insNum
        return tmp

    def build_instance(self, arr):
        attr = [arr[i] for i in self.attrSelected]
        tar = [arr[i] for i in self.tarSelected]
        return instance(attr, tar)

    def get_colTar(self, line):
        titles = line.split(',')
        colSel=[] #Col-Selected could be multiple indices
        for col in self.attrSelected:
            if col < len(titles) and col >= 0:
                colSel.append(titles[col])
            else:
                raise formatException('Index Out Of Range')

        '''tarSelected = -1 means no targets found'''
        if self.tarSelected < len(titles) and self.tarSelected >= 0:
            tarSel=titles[self.tarSelected]
        elif self.tarSelected == -1:
            tarSel = None
        else:
            raise formatException('Index Out Of Range')
        return (colSel, tarSel)

    def close_file(self):
        self.dataSource.close()

    def get_attrName(self):
        return self.attrNameVec

    def get_tarName(self):
        return self.targetNameVec
