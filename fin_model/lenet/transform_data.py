from ether.instance.format.mnist import readMnist
import numpy as np

source = r'E:\VirtualDesktop\nnet\minist\mnist.pkl.gz'
outPath = r'E:\VirtualDesktop\nnet\minist\normed_double_mnist.pkl.gz'
picPath = r'E:\VirtualDesktop\nnet\minist\normed_double_mnist.pkl'

xData = readMnist( source )

yData00 = np.array(xData[0][0], dtype=np.double)/256. #data are transformed to double
yData01 = np.array(xData[0][1]) #class label remains as int
yData10 = np.array(xData[1][0], dtype=np.double) /256.
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
