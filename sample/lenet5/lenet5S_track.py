from nnet import *
import pickle

debug = True
picPath = r'E:\VirtualDesktop\lenet\kernels.pkl'
filePath = r'E:\VirtualDesktop\lenet\double_mnist.pkl.gz'

#Preparation
mnist_reader = mnistDataReader(filePath, 10)
ff = picFilter()
db = filterPool( mnist_reader, ff )

#init tracker
opt = SGDOptimizer(lenet)
ltracker = layerTracker(opt)
classifyVal = classifyValidator(argmin)

with open(picPath, 'rb') as f:
    kernels = pickle.load(f)

#Buiding net
input_layer = inputLayer( (28, 28) )
C1 = [] #convolution 2D layer
for i in xrange(6):
    C1.append( conv2DLayer( (28, 28) ) )
    C1[i].connect(input_layer)

merge = merge1DLayer()
merge.connect( *C1 )
#tanhl1 = tanhLayer(1.7, 2./3.)
#tanhl1.connect(merge)

weight_layer = weightLayer(70)
weight_layer.connect(merge)
tanhl2 = tanhLayer(1.7, 2./3.)
tanhl2.connect(weight_layer)

rbf_layers = []
for i in xrange(10):
    rbf_layers.append( fixedGassinRBFLayer(kernels[i]) )
    rbf_layers[i].connect(tanhl2)

merge2 = merge1DLayer()
merge2.connect(*rbf_layers)
softmax_layer = softmaxLayer()
softmax_layer.connect(merge2)

layers = []
layers.append( input_layer )
layers.extend( C1 )
layers.extend( [merge, weight_layer, tanhl2] )
layers.extend(rbf_layers)
layers.extend([merge2, softmax_layer])


#Varifying shapes
if debug:
    print 'input_layer', input_layer.get_outputShape()
    print 'C1', C1[0].get_outputShape()
    print 'merge', merge.get_outputShape()
    #print 'tanh', tanhl1.get_outputShape()
    print 'weight_layer', weight_layer.get_outputShape()
    print 'tanh', tanhl2.get_outputShape()
    print 'rbf_layer', rbf_layers[0].get_outputShape()
    print 'merge2', merge2.get_outputShape()
    print 'softmax', softmax_layer.get_outputShape()
for layer in layers:
    layer.verify_shape()

#Put together
n_net = nnet(layers)
print 'building trainer'
ltracker.add_trackLayers(merge,weight_layer, tanhl2, merge2, softmax_layer)
#ltracker.set_params(weight_layer.get_bias(), weight_layer.get_weights())
tri = trainer(db, ltracker, classifyVal, n_net)
#Add layers to layer_tracker

#training
print 'training start'
tri.train(40000)

#Dumping nnet
try:
    with open(r'E:\VirtualDesktop\lenet\nnet.pkl', 'wb') as fi:
        pickle.dump(n_net, fi)
except Exception:
    print 'Exception occured during the process of picking'

#Validate
print 'validating'
print(tri.validate(200))

ltracker.print_info(maxCycles=20)
