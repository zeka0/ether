from ether.instance.format.mnist import readMnist
import numpy as np

def norm_img(img):
    boolArr = (img == 0)
    img[boolArr] = -1
    boolArr = (boolArr == False) #reverse the boolArr
    img[boolArr] = 1
    return img

def trim_img(img):
    mat = []
    for row in img:
        mat.append(row[4:-3])
    return np.array(mat)

filePath = r'E:\VirtualDesktop\nnet\minist\double_mnist.pkl.gz'
picPath = r'E:\VirtualDesktop\nnet\minist\kernels.pkl'

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
tmpImg = None
tmp2=[]

for img in yData:
    tmpImg = fun(img)
    tmpImg = norm_img(tmpImg)[2:-2]
    tmp.append(trim_img(tmpImg))

for img in tmp:
    tmp2.append(img.flat[:])

yyData = tuple(tmp2)

try:
    import pickle
    fi = open(picPath, 'wb')
    pickle.dump(yyData, fi)
    fi.close()
except Exception:
    print 'Exception occured during the process of picking'
    pass

#ultimate size of the yyData is (1, 70)
