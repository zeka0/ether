import pickle
import gzip
import numpy as np

def readMnist(filePath):
    with gzip.open( filePath, 'rb' ) as f:
        x = pickle.load(f)
        return x

if __name__ == '__main__':
    x = readMnist( r'E:\VirtualDesktop\mnist.pkl.gz' )
    x_train = x[0]
    print len(x_train)
    print x_train[1]
    print len( x_train[0] )
    print x_train[0].shape
    y = np.vstack( (x[0][0], x[1][0]) )
    print len( y )
    print y[0].shape
    print y[1].shape
    import pprint
    pprint.pprint(y)
