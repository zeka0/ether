import pickle

pkl_file = open(r'E:\VirtualDesktop\lenet\nnet.pkl', 'rb')
nnet = pickle.load(pkl_file)
pkl_file.close()

print 'end'
