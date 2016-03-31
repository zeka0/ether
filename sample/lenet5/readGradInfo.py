import pickle
fi = open(r'E:\VirtualDesktop\lenet\tracker.pkl', 'rb')
xl = pickle.load(fi)
xl.print_info()
