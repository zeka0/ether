import pickle
fi = open(r'E:\VirtualDesktop\layerTracker.pkl', 'rb')
xl = pickle.load(fi)
xl.print_info()
