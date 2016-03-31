from nnet.instance.format.mnist import readMnist
import numpy as np

def norm_img(img):
    boolArr = (img == 0)
    img[boolArr] = -1
    boolArr = (boolArr == False) #reverse the boolArr
    img[boolArr] = 1
    return img

filePath = r'E:\VirtualDesktop\lenet\double_mnist.pkl.gz'
picPath = r'E:\VirtualDesktop\lenet\kernels.pkl'

xData = readMnist( filePath )
xData10 = xData[1][0]

yData = (xData10[3], xData10[2], xData10[1], xData10[44], xData10[4],
         xData10[8], xData10[11], xData10[0], xData10[61], xData10[62])

from theano.tensor.signal.conv import conv2d
import theano
from theano import tensor as T
xT = T.matrix()
filter = np.ones( (1, 1) )
outputTensor = conv2d(input=xT, filters=filter, subsample=(2,2) )
fun = theano.function(inputs=[xT], outputs=outputTensor)

tmp = []
for img in yData:
    tmp.append(img[1:26][6:22])
tmp2 = []
tmpImg = None
for img in tmp:
    tmpImg = fun(img)
    tmp2.append(norm_img(tmpImg).flat[:])
yyData = tuple(tmp2)

try:
    import pickle
    fi = open(picPath, 'wb')
    pickle.dump(yyData, fi)
    fi.close()
except Exception:
    print 'Exception occured during the process of picking'
    pass
