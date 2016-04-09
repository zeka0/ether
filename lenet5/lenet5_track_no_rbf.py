from nnet import *
import pickle

debug = True
filePath = r'E:\VirtualDesktop\lenet\double_mnist.pkl.gz'

#Preparation
mnist_reader = mnistDataReader(filePath, 10)
ff = picFilter()
db = filterPool( mnist_reader, ff )

#init tracker
opt = SGDOptimizer(lenet)
ltracker = layerTracker(opt)
classifyVal = classifyValidator(argmin)

#Buiding net
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
for i in xrange(40):
    C5.append( conv2DLayer( (4, 4) ) )
    C5[i].connect( *S4 )

merge_layer = merge1DLayer()
merge_layer.connect( *C5 )

weight_layer = weightLayer(10)
weight_layer.connect(merge_layer)

tanh_layer = tanhLayer(1.7, 2./3.)
tanh_layer.connect(weight_layer)

softmax_layer = softmaxLayer()
softmax_layer.connect(weight_layer)

layers = []
layers.append( input_layer )
layers.extend( C1 )
layers.extend( S2 )
layers.extend( C3 )
layers.extend( S4 )
layers.extend( C5 )
layers.extend( [merge_layer, weight_layer, tanh_layer] )
layers.extend([softmax_layer])

#Varifying shapes
if debug:
    print 'input_layer', input_layer.get_outputShape()
    print 'C1', C1[0].get_outputShape()
    print 'S2', S2[0].get_outputShape()
    print 'C3', C3[0].get_outputShape()
    print 'S4', S4[0].get_outputShape()
    print 'C5', C5[0].get_outputShape()
    print 'merge', merge_layer.get_outputShape()
    print 'weight', weight_layer.get_outputShape()
    print 'tanh', tanh_layer.get_outputShape()
    print 'softmax', softmax_layer.get_outputShape()
for layer in layers:
    layer.verify_shape()

#Put together
n_net = nnet(layers)
print 'building trainer'
ltracker.add_trackLayers(merge_layer, weight_layer, tanh_layer, softmax_layer)
tri = trainer(db, ltracker, classifyVal, n_net)
#Add layers to layer_tracker

#training
print 'training start'
tri.train(2000)

#Dumping nnet
try:
    with open(r'E:\VirtualDesktop\lenet\nnet.pkl', 'wb') as fi:
        pickle.dump(n_net, fi)
except Exception:
    print 'Exception occured during the process of picking nnet'
try:
    with open(r'E:\VirtualDesktop\lenet\tri.pkl', 'wb') as fi:
        pickle.dump(tri, fi)
except Exception:
    print 'Exception occured during the process of picking of trainer'

#Validate
print 'validating'
print(tri.validate(200))

ltracker.print_info(maxCycles=20)
