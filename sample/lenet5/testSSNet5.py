from nnet import *

debug = True

'''
The mnist I'm using only has 10 classes (0-9)
And the image size is (28L, 28L)
So this will be a little different from the orignal LeNet-5
'''

mnist_reader = mnistDataReader( r'E:\VirtualDesktop\mnist.pkl.gz', 10)
ff = picFilter()
db = filterPool( mnist_reader, ff )
classifyVal = classifyValidator()
opt = crossEntro_SGDOptimizer() #rubbish class label required

input_layer = inputLayer( (28, 28) )
C1 = [] #convolution 2D layer
for i in xrange(10):
    C1.append( conv2DLayer( (28, 28) ) )
    C1[i].connect(input_layer)
merge = merge1DLayer()
merge.connect(*C1)
layers = []
layers.append( input_layer )
layers.extend( C1 )
layers.append(merge)

for layer in layers:
    layer.verify_shape()

n_net = nnet(layers)
tri = trainer(db, opt, classifyVal, n_net)

print 'training start'
tri.train(0)
try:
    import pickle
    fi = open(r'E:\VirtualDesktop\nnet.pkl', 'wb')
    pickle.dump(n_net, fi)
    fi.close()
except Exception:
    print 'Exception occured during the process of picking'
    pass

print 'validating'
print(tri.validate(200))
