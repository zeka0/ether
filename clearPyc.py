import os
import sys

rootDir = '.'

for dirPath, dirNames, fileNames in os.walk(rootDir):
    for fileName in fileNames:
        if fileName.endswith('.pyc'):
            os.remove( os.path.join(dirPath, fileName) )
