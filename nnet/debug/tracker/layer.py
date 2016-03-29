from core import tracker

class layerTracker(tracker):
    def __init__(self, opt):
        tracker.__init__(self, opt)
        self.layerOutputs = dict()

    def get_trackKeys(self):
        if not hasattr(self, 'trackLayers'):
            self.trackLayers = []
        return self.trackLayers

    def add_track_layers(self, *layers):
        for layer in layers:
            assert layer in self.get_layers()
        self.get_trackKeys().extend(layers)

    def init_track(self):
        track_list = []
        for layer in self.get_trackKeys():
            self.layerOutputs[layer] = []
            track_list.append(layer.get_outputTensor)
        self.optimizer.add_additionalOutputs(track_list)

    def train_once(self, attr, tar):
        result = self.optimizer.train_once(attr, tar)
        for layer, out in zip(self.layerOutputs.keys(), result):
            self.track(layer, out)

    def print_info(self, maxCycles=3):
        for i in xrange(3):
            print 'Ouputs in cycle ', i
            for layer in self.get_trackKeys():
                print 'Layer: ', layer
                print self.layerOutputs[layer][i]
