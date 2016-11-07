from ether import *

debug = True
filePath = r'E:\VirtualDesktop\nnet\minist\normed_double_mnist.pkl.gz'
model_fname = 'lenet'

mnist_reader = mnistDataReader(filePath, 10)
db = fullInstancePool(mnist_reader.read_all(), True)
ff = dimFilter((1, 1, 28, 28))
db = filterPool(db, ff)
classifyVal = classifyValidator(argmax)
opt = SGDOptimizer()

biasInitDic = {'distr':'constant', 'value':0.}
weightBiasInitDic = {'distr':'constant', 'value':0}

input_layer = inputLayer( (1, 1, 28, 28) )
C1 = conv2DLayer(4, (5, 5), border_mode='valid', filter={'distr':'uniform', 'low':-np.sqrt( 6./125 ), 'high':np.sqrt( 6./125 )}, bias=biasInitDic)
M2 = maxPoolLayer((2,2), ignore_border=True)
T1 = tanhLayer(1, 1)
C3 = conv2DLayer(6, (5, 5), border_mode='valid', filter={'distr':'uniform', 'low':-np.sqrt( 6./150 ), 'high':np.sqrt( 6./150 )}, bias=biasInitDic)
M4 = maxPoolLayer((2,2), ignore_border=True)
T2 = tanhLayer(1, 1)
F5 = flattenLayer()
W6 = weightLayer(500, weight={'distr':'uniform', 'low':-np.sqrt( 12./596 ), 'high':np.sqrt( 12./596 )}, bias=weightBiasInitDic)
DP1 = dropoutLayer(0.5)

T3 = tanhLayer(1, 1)
W7 = weightLayer(10, weight={'distr':'uniform', 'low':-np.sqrt( 12./510 ), 'high':np.sqrt( 12./510 )}, bias=weightBiasInitDic)
DP2 = dropoutLayer(0.5)

S8 = softmaxLayer()
layers = [input_layer, C1, M2, T1, C3, M4, T2, F5, W6, DP1, T3, W7, DP2, S8]
for i in xrange(len(layers) - 1):
    layers[i + 1].connect(layers[i])
for l in layers:
    print l.get_outputShape()
#Put together
n_net = nnet(layers, cost_func=negtive_log, monitor_cost_func=None)
train_ep = DropoutResetEp(DP1, DP2)
tri = trainer(db, opt, classifyVal, n_net, train_ep=train_ep)

print 'compling the trainer'
tri.compile()
#training
print 'training start'
tri.train(40000)

dump_trainer(tri)

#Validate
print 'validating'
print 'canceling dropout effect'
DP1.cancel_bitvec()
DP2.cancel_bitvec()
print(tri.validate(200))
