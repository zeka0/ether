from nnet import *
import pickle

debug = True
picPath = r'E:\VirtualDesktop\lenet\kernels.pkl'
filePath = r'E:\VirtualDesktop\lenet\double_mnist.pkl.gz'

#Preparation
mnist_reader = mnistDataReader(filePath, 10)
ff = picFilter(grey_num=-1, white_num=1)
db = filterPool( mnist_reader, ff )
classifyVal = classifyValidator(argmin)
opt = SGDOptimizer(lenet)

with open(picPath, 'rb') as f:
    kernels = pickle.load(f)

filterInitDic = {'distr':'normal', 'std':1e-2, 'mean':0}
biasInitDic = {'distr':'scala', 'type':float, 'value':0}
weightInitDic = {'distr':'uniform', 'low':-0.25, 'high':0.25}
coefInitDic = {'distr':'scala', 'type':float, 'value':1}

#Buiding net
input_layer = inputLayer( (28, 28) )
C1 = [] #convolution 2D layer
for i in xrange(6):
    C1.append( conv2DLayer( (5, 5), bias=biasInitDic, filter=filterInitDic) )
    C1[i].connect(input_layer)
S2 = [] #subsample layer
for i in xrange(6):
    S2.append( subSampleLayer( (2, 2), bias=biasInitDic, coef=coefInitDic ) )
    S2[i].connect( C1[i] )
C3 = []
for i in xrange(16):
    C3.append( conv2DLayer( (5, 5), bias=biasInitDic, filter=filterInitDic ) )
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
    S4.append( subSampleLayer( (2, 2), bias=biasInitDic, coef=coefInitDic ) )
for i in xrange(16):
    S4[i].connect( C3[i] )
C5 = []

#TODO: change number of the conv2DLayer
for i in xrange(40):
    C5.append( conv2DLayer( (4, 4), bias=biasInitDic, filter=filterInitDic ) )
    C5[i].connect( *S4 )

merge_layer = merge1DLayer()
merge_layer.connect( *C5 )

weight_layer = weightLayer(70, bias=biasInitDic, weight=weightInitDic)
weight_layer.connect(merge_layer)

tanh_layer = tanhLayer(1.7, 2./3.)
tanh_layer.connect(weight_layer)

rbf_layers = []
for i in xrange(10):
    rbf_layers.append( fixedRBFLayer(gausi_rbf, kernels[i]) )
    rbf_layers[i].connect(tanh_layer)

merge_layer2 = merge1DLayer()
merge_layer2.connect(*rbf_layers)
softmax_layer = softmaxLayer()
softmax_layer.connect(merge_layer2)

layers = []
layers.append( input_layer )
layers.extend( C1 )
layers.extend( S2 )
layers.extend( C3 )
layers.extend( S4 )
layers.extend( C5 )
layers.extend( [merge_layer, weight_layer, tanh_layer] )
layers.extend(rbf_layers)
layers.extend([merge_layer2, softmax_layer])

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
    print 'rbf_layer', rbf_layers[0].get_outputShape()
    print 'merge2', merge_layer2.get_outputShape()
    print 'softmax', softmax_layer.get_outputShape()
for layer in layers:
    layer.verify_shape()

#Put together
n_net = nnet(layers)
tri = trainer(db, opt, classifyVal, n_net)

print 'compling the trainer'
tri.compile()
#training
print 'training start'
tri.train(10000)

dump_trainer(tri)

#Validate
print 'validating'
print(tri.validate(200))
