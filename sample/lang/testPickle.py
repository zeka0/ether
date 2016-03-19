import pickle

li = [1, 2, 3]
fi = open(r'E:\VirtualDesktop\list.pkl','wb')
pickle.dump(li, fi)
fi.close()