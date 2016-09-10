from ether import *

debug = True
filePath = r'E:\VirtualDesktop\nnet\csv\reddit-comments-2015-08.csv'
model_fname = 'nltk-rnn'

csv_reader = csvDataReader(filePath, maxQueueLen=100, colSelected=[0])
db = fullPool(csv_reader.read_all(), True)
ff = nltkFilter()
db = filterPool(db, ff)
classifyVal = classifyValidator(argmax)
opt = SGDOptimizer()

biasInitDic = {'distr':'constant', 'value':0.}
weightBiasInitDic = {'distr':'constant', 'value':0}

#weight={'distr':'uniform', 'low':-np.sqrt( 6./8100 ), 'high':np.sqrt( 6./8100 )}
input_layer = inputLayer( (1, 8000) )
R1 = recurrentLayer(100, U = {'distr':'uniform', 'low':-np.sqrt( 6./8100 ), 'high':np.sqrt( 6./8100 )},
                    W={'distr':'uniform', 'low':-np.sqrt( 6./200), 'high':np.sqrt( 6./200)},
                    V={'distr':'uniform', 'low':-np.sqrt( 6./8100 ), 'high':np.sqrt( 6./8100 )})
S2 = softmaxLayer()
layers = [input_layer, R1, S8]
for i in xrange(len(layers) - 1):
    layers[i + 1].connect(layers[i])
for l in layers:
    print l.get_outputShape()
#Put together
n_net = nnet(layers, cost_func=negtive_log, monitor_cost_func=None)
tri = trainer(db, opt, classifyVal, n_net)

print 'compling the trainer'
tri.compile()
#training
print 'training start'
tri.train(40000)

dump_trainer(tri)

#Validate
print 'validating'
print(tri.validate(200))
