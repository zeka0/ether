import pickle
fi = open(r'E:\VirtualDesktop\tracker.pkl', 'rb')
xl = pickle.load(fi)
xl.print_info()
