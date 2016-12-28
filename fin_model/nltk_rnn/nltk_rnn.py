from ether import *

debug = True
filePath = r'E:\VirtualDesktop\nnet\csv\reddit-comments-2015-08.csv'
model_fname = 'nltk-rnn'

'''Execute build_datapool.py before executing this file'''
classifyVal = classifyValidator(argmax)
opt = SGDOptimizer()

biasInitDic = {'distr':'constant', 'value':0.}
weightBiasInitDic = {'distr':'constant', 'value':0}

#weight={'distr':'uniform', 'low':-np.sqrt( 6./8100 ), 'high':np.sqrt( 6./8100 )}

#since rnn requires the input to be of any length

input_layer = inputLayer( (-1, -1, 1) )#8000 to make other layers know what they are dealing with
R1 = recurrentLayer(100, U = {'distr':'uniform', 'low':-np.sqrt( 6./8100 ), 'high':np.sqrt( 6./8100 )},
                    W={'distr':'uniform', 'low':-np.sqrt( 6./200), 'high':np.sqrt( 6./200)},
                    V={'distr':'uniform', 'low':-np.sqrt( 6./8100 ), 'high':np.sqrt( 6./8100 )})
layers = [input_layer, R1]
for i in xrange(len(layers) - 1):
    layers[i + 1].connect(layers[i])
for l in layers:
    print_shape(l.get_outputShape())
#Put together
n_net = nnet(layers, cost_func=cross_entro, monitor_cost_func=None)

db = load_pool() #lazy load
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
