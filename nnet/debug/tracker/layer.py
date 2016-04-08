from core import tracker

class layerTracker(tracker):
    def __init__(self, opt):
        tracker.__init__(self, opt)
        self.trackLayers = []

    def get_trackKeys(self):
        return self.trackLayers

    def add_trackLayers(self, *layers):
        self.trackLayers.extend(layers)

    def init_track(self):
        track_list = []
        for layer in self.get_trackKeys():
            track_list.append(layer.get_outputTensor())
        self.optimizer.add_additionalOutputs(*track_list)
