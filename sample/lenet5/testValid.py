import pickle

pkl_file = open(r'E:\VirtualDesktop\nnet.pkl', 'rb')
net = pickle.load(pkl_file)

print 'end'
