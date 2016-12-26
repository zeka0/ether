from ether import *

#TODO has many problems
debug = True
#filePath = r'E:\VirtualDesktop\nnet\minist\normed_double_mnist.pkl.gz'
filePath = r'E:\VirtualDesktop\nnet\minist\flatten_double_mnist.pkl.gz'
model_fname = 'lenet'

ff = dimFilter((1, 1, 28, 28), (1, 10))
classifyVal = classifyValidator(argmax)
opt = HessianFreeOptimizer()

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
W6 = weightLayer(500, weight={'distr':'uniform', 'low':-np.sqrt( 6./596 ), 'high':np.sqrt( 6./596 )}, bias=weightBiasInitDic)
T3 = tanhLayer(1, 1)
W7 = weightLayer(10, weight={'distr':'uniform', 'low':-np.sqrt( 6./510 ), 'high':np.sqrt( 6./510 )}, bias=weightBiasInitDic)
S8 = softmaxLayer()
layers = [input_layer, C1, M2, T1, C3, M4, T2, F5, W6, T3, W7, S8]
for i in xrange(len(layers) - 1):
    layers[i + 1].connect(layers[i])
for l in layers:
    print l.get_outputShape()
#Put together
mnist_reader = mnistDataReader(filePath, 10)
db = fullInstancePool(mnist_reader.read_all(), True)
#db = fullPool(None, True)
db = filterPool(db, [ff])
n_net = nnet(layers, cost_func=negtive_log, monitor_cost_func=None)
tri = trainer(db, opt, classifyVal, n_net)

print 'compling the trainer'
tri.compile()
#training
print 'training start'
tri.train(2000)

dump_trainer(tri)

#Validate
print 'validating'
print(tri.validate(200))
