import pickle
from ether import *

def load_model(path):
    try:
        with open( path, 'rb' ) as fi:
            return pickle.load(fi)
    except Exception as ex:
        print 'Exception occured in process of loading model'
        print ex


if __name__ == '__main__':
    path = r'E:\VirtualDesktop\nnet\model\rbm.pkl'
    rbm = load_model(path)
    filePath = r'E:\VirtualDesktop\nnet\minist\flatten_double_mnist.pkl.gz'
    mnist_reader = mnistDataReader(filePath, 10)
    db = fullPool(mnist_reader.read_all(), True)

    import PIL.Image as Image
    image_data = np.zeros(
        (29 * 20 + 1, 28),
        dtype='uint8'
    )
    from ether.debug.plot import tile_raster_images
    print 'sampling from rbm'
    for i in xrange(20):
        ins = db.read_instances(1)[0]
        vis_mf = rbm.feed_forward( ins.get_attr() )
        print 'sampled image of ', np.argmax(ins.get_target())
        image_data[29 * i: 29 * i + 28, :] = tile_raster_images(
            X=vis_mf,
            img_shape=(28, 28),
            tile_shape=(1, 1),
            tile_spacing=(1, 1)
        )

    print 'saving image'
    image = Image.fromarray(image_data)
    image.save( r'E:\VirtualDesktop\nnet\minist\sample.png' )
    import pickle
    try:
        with open( r'E:\VirtualDesktop\nnet\model\rbm.pkl', 'wb' ) as fi:
            pickle.dump(rbm, fi)
    except Exception as ex:
        print 'Exception occured in process of dumping model'
        print ex
