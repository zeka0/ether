from ether.instance.format.mnist import readMnist
import numpy as np

source = r'E:\VirtualDesktop\nnet\minist\normed_double_mnist.pkl.gz'
outPath = r'E:\VirtualDesktop\nnet\minist\flatten_double_mnist.pkl.gz'
picPath = r'E:\VirtualDesktop\nnet\minist\flatten_double_mnist.pkl'

xData = readMnist( source )

yData00 = []
yData10 = []
for i in xrange(len(xData[0][0])):
    yData00.append(xData[0][0][i].flat[:])
for i in xrange(len(xData[1][0])):
    yData10.append(xData[1][0][i].flat[:])

yData = ((np.array(yData00), xData[0][1]), (np.array(yData10), xData[1][1]))

try:
    import pickle
    fi = open(picPath, 'wb')
    pickle.dump(yData, fi)
    fi.close()
except Exception:
    print 'Exception occured during the process of picking'
    pass

import gzip
import shutil
with open(picPath, 'rb') as f_in, gzip.open(outPath, 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
