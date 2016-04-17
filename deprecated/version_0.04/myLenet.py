from nnet import *
import pickle

debug = True
track = False
valid_print = False
picPath = r'E:\VirtualDesktop\lenet\kernels.pkl'
filePath = r'E:\VirtualDesktop\lenet\double_mnist.pkl.gz'

mnist_reader = mnistDataReader(filePath, 10)
ff = picFilter(grey_num= 1, white_num= -1)
fulldb = fullPool(mnist_reader.read_all(), True)
db = filterPool( fulldb, ff )

with open(picPath, 'rb') as f:
    kernels = pickle.load(f)

filterInitDic = {'distr':'normal', 'std':1e-1, 'mean':0}
biasInitDic = {'distr':'scala', 'type':float, 'value':0}
coefInitDic = {'distr':'scala', 'type':float, 'value':1}

input_layer = inputLayer( (28, 28) )
C1 = []
for i in xrange(6):
    C1.append( conv2DLayer( (5, 5), bias=biasInitDic, filter=filterInitDic ) )
    C1[i].connect(input_layer)

S2 = []
for i in xrange(6):
    S2.append( subSampleLayer((2, 2), bias=biasInitDic, coef=coefInitDic) )
    S2[i].connect(C1[i])

C3 = []
for i in xrange(10):
    C3.append( conv2DLayer( (3, 6), bias=biasInitDic, filter=filterInitDic ) )
    C3[i].connect( *S2 )

T4 = []
for i in xrange(10):
    T4.append( tanhLayer(1.7, 2./3.) )
    T4[i].connect(C3[i])

F5 = []
for i in xrange(10):
    F5.append( flattenLayer() )
    F5[i].connect(T4[i])

R6 = []
for i in xrange(10):
    R6.append( fixedRBFLayer(gausi_rbf, kernels[i]) )
    R6[i].connect(F5[i])

M7 = merge1DLayer()
M7.connect( *R6 )
S8 = softmaxLayer()
S8.connect( M7)

layers = [input_layer]
layers.extend( C1 )
layers.extend( S2 )
layers.extend( C3 )
layers.extend( T4 )
layers.extend( F5 )
layers.extend( R6 )
layers.extend( [M7, S8] )

if debug:
    print 'input_layer', input_layer.get_outputShape()
    print 'C1', C1[0].get_outputShape()
    print 'S2', S2[0].get_outputShape()
    print 'C3', C3[0].get_outputShape()
    print 'T4', T4[0].get_outputShape()
    print 'F5', F5[0].get_outputShape()
    print 'R6', R6[0].get_outputShape()
    print 'M7', M7.get_outputShape()
    print 'S8', S8.get_outputShape()
for layer in layers:
    layer.verify_shape()

n_net = nnet(layers)
classifyVal = classifyValidator(argmin)
opt = SGDOptimizer(lenet)
if track:
    trk = layerTracker(opt)
    trk.add_trackLayers( C3[0], T4[0], F5[0], M7, S8 )
    tri = trainer(db, trk, classifyVal, n_net)
else:
    tri = trainer(db, opt, classifyVal, n_net)

print 'Compling the trainer'
tri.compile()
print 'Training start'
tri.train(80000)

nnet_fname = 'myLenet'
optimizer_fname = 'myOpt'
validator_fname = 'myValid'

if track:
    print 'Tracker printing'
    trk.print_info( maxCycles=20 )

print 'Dumping Nnet'
dump_trainer(tri)

print 'Validating'
print(tri.validate(2000))
