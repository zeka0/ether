import idx2numpy

train_img_path = r'E:\VirtualDesktop\lenet\train-images.idx3-ubyte'
train_index_path = r'E:\VirtualDesktop\lenet\train-labels.idx1-ubyte'
t10k_img_path = r'E:\VirtualDesktop\lenet\t10k-images.idx3-ubyte'
t10k_index_path = r'E:\VirtualDesktop\lenet\t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_img_path)
train_labels = idx2numpy.convert_from_file(train_index_path)

print 'End'