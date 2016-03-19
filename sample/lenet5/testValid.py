from nnet import *
import pickle

classifyVal = classifyValidator()
pkl_file = open(r'E:\VirtualDesktop\nnet.pkl', 'rb')
net = pickle.load(pkl_file)
net.set_validator(classifyVal)

print 'validating'
print(classifyVal.validate(maxCycles=200))
