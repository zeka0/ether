from ether import *

print 'init scanner'
scanner = cvsScanner(r'/Users/mac/Desktop/REF_PLANT_DICTIONARY.csv', headLength=0, hasTitle=True)
dic = scanner.scan_all()
scanner.close_file()

print 'init classList'
classList = []
#classList.append(dic[1])
classList.append(dic[7])
classList.append(dic[19])

tarList = dic[17] #Family
print 'init value interpreter'
valInter = multi_targetInterpreter(classList)

print 'init target interpreter'
tarInter = targetInterpreter(tarList)

print 'init reader'
#Prediction of species, SYMBOL & CATEGORY & GENUS to predict FAMILY
reader = cvsDataReader(tarInter, valInter, 100,
                       r'/Users/mac/Desktop/REF_PLANT_DICTIONARY.csv', [7, 19], 17, hasTitle=True)

print 'init dataBase'
db = dataBase(core, True, filter())

print 'init stochastic dataBase'
db.stochastic_init()

print 'creating validator and optimizer'
val=classifyValidator()
opt=sqrtMeanOptimizer()

print 'init neural network'
nnet = nnet([8, 5, 2, 3], [T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.softmax],
            T.nnet.softmax, T.nnet.sigmoid, db, opt, val)
#nnet = nnet([7, 5, 4], [T.nnet.sigmoid, T.nnet.sigmoid, T.nnet.sigmoid], T.nnet.sigmoid, T.nnet.sigmoid, db, opt, val)

print 'training start'
opt.train(200)
#opt.train(100)

print 'validating'
print(val.validate(maxCycles=200))
core.close_file()