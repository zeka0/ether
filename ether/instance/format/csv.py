import os
from collections import deque

from ether.instance.instance import instance
from ether.util import *
from core import dataReader

class csvDataReader(dataReader):
    '''
    hasTitle specifies whether the cvs file has title attributes
    colSelected is a list specifying which column should be used
    targetSelected is a list specifing which column should be treated as targetValue
    '''
    def __init__(self, filePath,
                 maxQueueLen, colSelected, targetSelected=-1,
                 headLength=0, hasTitle=False):
        '''
        colSelected should be a list
        targetSelected should specify a column number
        targetSelected == -1 means no targets provided
        maxLen is the maxium of the readable instance
        '''
        self.instanceBuff = deque(maxlen=maxQueueLen)
        self.dataSource=open(filePath, 'rU') #Read Universial
        self.__csv_init__(headLength, hasTitle, colSelected, targetSelected)

    def __select_colTar__(self, line):
        titles = line.split(',')
        colSel=[] #Col-Selected could be multiple indices
        for col in self.colSelected:
            if col < len(titles) and col >= 0:
                colSel.append(titles[col])
            else:
                raise formatException('Index Out Of Range')

        '''targetSelected = -1 means no targets found'''
        if self.targetSelected < len(titles) and self.targetSelected >= 0:
            tarSel=titles[self.targetSelected]
        elif self.targetSelected == -1:
            tarSel = None
        else:
            raise formatException('Index Out Of Range')

        return (colSel, tarSel)

    def __csv_init__(self, headLength, hasTitle, colSelected, targetSelected):
        '''
        Parsing and recording relevant infomation
        '''
        for i in xrange(0, headLength): #Discard lines
            self.dataSource.readline()
        self.colSelected=colSelected
        self.targetSelected=targetSelected
        if hasTitle:
            line = self.dataSource.readline()
            colSel, tarSel = self.__select_colTar__(line)
            self.set_attrVec(colSel, tarSel)

    def __read_source__(self, size): #Generator
        for i in xrange(0, size):
            line = self.dataSource.readline()
            if line is not None:
                colSel, tarSel = self.__select_colTar__(line)
                yield instance(colSel, tarSel)
            else: break

    def has_nextInstance(self, size):
        #TODO not efficient
        currIndex = self.dataSource.tell()
        lines = self.dataSource.readlines()
        self.dataSource.seek(currIndex, os.SEEK_SET)
        if len(lines) >= size:
            return False
        else: return True

    def close_file(self):
        self.dataSource.close()

    def read_all(self):
        tmp = []
        while self.has_nextInstance(1):
            for ins in self.read_instance(1):
                tmp.append(ins)
        return tmp

    def read_instance(self, batchSize=1): #Generator
        if batchSize> self.avail_sizeNow():
            extSize=batchSize-self.avail_sizeNow()
            self.instanceBuff.extend(self.__read_source__(extSize))
        for i in xrange(batchSize):
            yield self.instanceBuff.pop()

    def get_attrName(self):
        return self.attrNameVec

    def get_tarName(self):
        return self.targetNameVec

    def set_attrVec(self, attrNameVec, targetNameVec):
        self.attrNameVec=attrNameVec
        self.targetNameVec=targetNameVec

    def avail_sizeNow(self):
        return len(self.instanceBuff)

class csvScanner:
    '''
    Cvs scanner treats the first column in a cvs file has index '0'
    CvsScanner scans the whole cvs file to find classes that is required to make a interpreter
    '''
    def __init__(self, filepath, headLength=0, hasTitle=True):
        '''
        Caution: title isn't considered as head
        '''
        self.file = open(filepath, 'r+')
        self.headLength = headLength
        self.hasTitle = hasTitle

    def close_file(self):
        self.file.close()

    def adjust_pointer(self):
        '''
        making file points to an appriate position
        '''
        if self.file.tell() != 0:
            self.file.seek(0)
        if self.hasTitle:
            #discard title
            self.file.readline()
        for i in xrange(0, self.headLength): #Discard lines
            self.file.readline()

    def scan_all(self, maxLen=1000):
        self.adjust_pointer()
        line = self.file.readline()
        cols = line.split(',')
        colSelected = [i for i in xrange(len(cols))]
        return self.scan(colSelected)

    def scan(self, colSelected, maxLen=1000):
        '''
        return dictionary
        maxLen specifies maxLen of lines to read
        '''
        self.adjust_pointer()
        classDic=dict()
        #creating entries
        for i in xrange(len(colSelected)):
            classDic[colSelected[i]]=[]
        for line, k in zip(self.file.readlines(), xrange(maxLen)):
            cols = line.split(',')
            for i in xrange(len(colSelected)):
                colIndex=colSelected[i]
                if cols[colIndex] not in classDic[colIndex]:
                    classDic[colIndex].append(cols[colIndex])
        return classDic
