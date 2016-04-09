from nnet.instance.format.mnist import readMnist
import numpy as np

filePath = r'E:\VirtualDesktop\lenet\mnist.pkl.gz'
outPath = r'E:\VirtualDesktop\lenet\double_mnist.pkl.gz'
picPath = r'E:\VirtualDesktop\lenet\double_mnist.pkl'

xData = readMnist( filePath )

yData00 = np.array(xData[0][0], dtype=np.double) #data are transformed to double
yData01 = np.array(xData[0][1]) #class label remains as int
yData10 = np.array(xData[1][0], dtype=np.double)
yData11 = np.array(xData[1][1])
yData = ((yData00, yData01), (yData10, yData11))

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
