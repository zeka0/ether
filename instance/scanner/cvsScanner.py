__author__ = 'mac'
'''
cvsScanner scans the whole cvs file to find classes that is required to make a interpreter
'''
import warnings
import os

'''
cvs scanner treats the first column in a cvs file has index '0'
'''
class scanner:
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
