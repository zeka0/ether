import pickle
fi = open(r'E:\VirtualDesktop\lenet\layerTracker.pkl', 'rb')
xl = pickle.load(fi)
xl.print_info()
