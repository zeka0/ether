#After training, the conv2dLayer's filters and the biases are all 0
from nnet import *

debug = True

mnist_reader = mnistDataReader( r'E:\VirtualDesktop\mnist.pkl.gz', 10)
ff = picFilter()
db = filterPool(mnist_reader,ff)

db.drain_reader()
classifyVal = classifyValidator()
opt = crossEntro_SGDOptimizer() #rubbish class label required

input_layer = inputLayer( (28, 28) )
C1 = [] #convolution 2D layer
for i in xrange(6):
    C1.append( conv2DLayer( (5, 5) ) )
    C1[i].connect(input_layer)
S2 = [] #subsample layer
for i in xrange(6):
    S2.append( subSampleLayer( (2, 2) ) )
    S2[i].connect( C1[i] )
C3 = []
for i in xrange(16):
    C3.append( conv2DLayer( (5, 5) ) )
for i in xrange(6):
    C3[i].connect( S2[i], S2[ (i + 1) % 6 ], S2[ (i + 2)%6 ] )
for i in xrange(6):
    z = i + 6
    C3[z].connect( S2[ z % 6 ], S2[ (z + 1) % 6 ], S2[ (z + 2) % 6 ], S2[ (z + 3) % 6 ] )
C3[12].connect( S2[0], S2[1], S2[3], S2[4] )
C3[13].connect( S2[1], S2[2], S2[4], S2[5] )
C3[14].connect( S2[2], S2[3], S2[5], S2[1] )
C3[15].connect( *S2 )
S4 = []
for i in xrange(16):
    S4.append( subSampleLayer( (2, 2) ) )
for i in xrange(16):
    S4[i].connect( C3[i] )
C5 = []
for i in xrange(10):
    C5.append( conv2DLayer( (4, 4) ) )
    C5[i].connect( *S4 )

merge = merge1DLayer()
merge.connect( *C5 )
weight_layer = weightLayer(10)
weight_layer.connect( merge )

sftmax = softmaxLayer()
sftmax.connect(weight_layer)
#sftmax.connect(merge)

layers = []
layers.append( input_layer )
layers.extend( C1 )
layers.extend( S2 )
layers.extend( C3 )
layers.extend( S4 )
layers.extend( C5 )
layers.extend( [merge, weight_layer] )
layers.extend([sftmax])

if debug:
    print 'input_layer', input_layer.get_outputShape()
    print 'C1', C1[0].get_outputShape()
    print 'S2', S2[0].get_outputShape()
    print 'C3', C3[0].get_outputShape()
    print 'S4', S4[0].get_outputShape()
    print 'C5', C5[0].get_outputShape()
    print 'merge', merge.get_outputShape()
    print 'weight', weight_layer.get_outputShape()
    #print 'rbf_layer', rbf_layers[0].get_outputShape()
    #print 'merge2', merge2.get_outputShape()

for layer in layers:
    layer.verify_shape()

n_net = nnet(layers)
tri = trainer(db, opt, classifyVal, n_net)

print 'training start'
tri.train(100)

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