import pickle
with open(r'E:\VirtualDesktop\lenet\nnet.pkl', 'rb') as fi:
    net = pickle.load(fi)
print 'End'
