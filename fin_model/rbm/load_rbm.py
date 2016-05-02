import pickle

path = r'E:\VirtualDesktop\nnet\minist\rbm.pkl'

def load_model():
    try:
        with open( path, 'rb' ) as fi:
            return pickle.load(fi)
    except Exception as ex:
        print 'Exception occured in process of loading model'
        print ex

x = load_model()
print 'End'
