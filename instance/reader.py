__author__ = 'mac'
import instance
from collections import deque
from theanoNnet.nnetUtil.exception import *
import os

class dataReader(object):
    '''
    Interface+Buffering of the instances
    SubClass should implement the "read_source(self, size)" method
    '''
    def __init__(self, tarInter, valInter, maxLen=100):
        self.instanceBuff=deque(maxlen=maxLen)
        self.tarInter=tarInter
        self.valInter=valInter

    def get_bufferSize(self):
        return len(self.instanceBuff)

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

    def get_attrName(self):
        return self.attrNameVec

    def get_tarName(self):
        return self.targetNameVec

    def set_attrVec(self, attrNameVec, targetNameVec):
        self.attrNameVec=attrNameVec
        self.targetNameVec=targetNameVec

    def read_source(self, batchSize):
        '''
        When implementing this method, subClass should make the instanceBuff[begIndex:endIndex] contains only valid instances
        Other part of the instanceBuff can be invalidated instances
        This design is aimed to make dataReader has more flexibility
        '''
        raise NotImplementedError()

    def avail_sizeNow(self):
        return len(self.instanceBuff)


    def read_instance(self, batchSize=1): #Generator
        if batchSize> self.avail_sizeNow():
            extSize=batchSize-self.avail_sizeNow()
            self.instanceBuff.extend(self.read_source(extSize))
        for i in xrange(batchSize):
            yield self.instanceBuff.pop()

    def has_nextInstance(self, size):
        '''
        SubClass should implements this to tell if the reader can read extra 'size' instances
        '''
        raise NotImplementedError()

class cvsDataReader(dataReader):
    #hasTitle specifies whether the cvs file has title attributes
    #colSelected is a list specifying which column should be used
    #targetSelected is a list specifing which column should be treated as targetValue
    def __init__(self, tarInter, valInter, maxLen,
                 filePath, colSelected, targetSelected,
                 headLength=0, hasTitle=False):
        self.dataSource=open(filePath, 'rU') #Read Universial
        dataReader.__init__(self, tarInter, valInter, maxLen=maxLen)
        self.__cvs_init(headLength, hasTitle, colSelected, targetSelected)

    def __select_colTar(self, line):
        titles = line.split(',')
        colSel=[] #Col-Selected could be multiple indices
        for col in self.colSelected:
            if col < len(titles):
                colSel.append(titles[col])
            else:
                '''
                sometimes it means that your data format isn't very good
                you should be careful when this throws
                '''
                raise formatException('Index Out Of Range')

        if self.targetSelected < len(titles):
            tarSel=titles[self.targetSelected]
        else:
            raise formatException('Index Out Of Range')

        return (colSel, tarSel)

    def __cvs_init(self, headLength, hasTitle, colSelected, targetSelected):
        '''
        Parsing and recording relevant infomation
        '''
        for i in xrange(0, headLength): #Discard lines
            self.dataSource.readline()
        self.colSelected=colSelected
        self.targetSelected=targetSelected
        if hasTitle:
            line = self.dataSource.readline()
            colSel, tarSel = self.__select_colTar(line)
            self.set_attrVec(colSel, tarSel)

    def read_source(self, size): #Generator
        for i in xrange(0, size):
            line = self.dataSource.readline()
            if line is not None:
                colSel, tarSel = self.__select_colTar(line)
                #Use interpreter to interprete the values
                yield instance.instance(self.valInter.create_tuple(colSel),
                               self.tarInter.create_tuple(tarSel))
            else: break

    def has_nextInstance(self, size):
        currIndex = self.dataSource.tell()
        lines = self.dataSource.readlines()
        self.dataSource.seek(currIndex, os.SEEK_SET)
        if len(lines) >= size:
            return False
        else: return True

    '''
    it's vital to notice that the number of attributes can only be determined through interpreter
    '''
    def get_numOf_Attrs(self):
        return self.valInter.get_numOf_attrs()

    def get_numOf_Targets(self):
        return self.tarInter.get_numOf_attrs()

    def close_file(self):
        self.dataSource.close()
