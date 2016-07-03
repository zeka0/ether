from core import dataReader
import os
from ether.exterior_lib.midi.utils import midiread
import theano
class midiDataReader(dataReader):
    def __init__(self, dataPath, roll=(21, 109), speriod=0.3):
        '''
        :type roll: (integer, integer) tuple
        :param roll:   Specifies the pitch range of the piano-roll in MIDI note numbers,
                    including roll[0] but not roll[1], such that roll[1]-roll[0] is the number of
                    visible units of the RBM at a given time step. The default (21,
                    109) corresponds to the full range of piano (88 notes).
        :type speriod: float
        :param speriod: Sampling period when converting the MIDI files into piano-rolls, or
                    equivalently the time difference between consecutive time steps
        '''
        self.dataPath = dataPath
        self.roll = roll
        self.speriod = speriod
        assert os.path.isdir(dataPath)
        self.fpaths = [os.path.join(dataPath, fn) for fn in os.listdir(dataPath)]

    def get_numOf_Attrs(self):
        '''
        Return the number of attributes
        '''
        raise NotImplementedError()

    def get_numOf_Targets(self):
        '''
        Return the number of targets
        '''
        raise NotImplementedError()

    def read_instance(self, batchSize):
        '''
        It's required to return a list
        And returning a generator is also encouraged
        '''
        raise NotImplementedError()

    def has_nextInstance(self, size):
        '''
        SubClass should implements this to tell if the reader can read extra 'size' instances
        '''
        raise NotImplementedError()

    def read_all(self):
        '''
        Read all the instances availble
        '''
        dataset = [midiread(f, self.roll,
                            self.speriod).piano_roll.astype(theano.config.floatX)
                   for f in self.fpaths]
        return dataset
