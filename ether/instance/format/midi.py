from core import dataReader
import os
from ether.ext_lib.midi.utils import midiread
import theano
from ether.instance.instance import unlabledInstance, instance

class midiUnlabledDataReader(dataReader):
    def __init__(self, dataPath, roll=(21, 109), speriod=0.3):
        '''
        Currently only serves to be the unlabled data pool
        Subclass could use file-names of the data as the targets of the instances

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
        self.index = 0

    def get_numOf_Attrs(self):
        '''
        Return the number of attributes
        '''
        raise NotImplementedError('No attributes for midi files')

    def get_numOf_Targets(self):
        '''
        Return the number of targets
        '''
        raise NotImplementedError('No targets for midi files')

    def read_instance(self, batchSize):
        '''
        It's required to return a list
        And returning a generator is also encouraged
        '''
        for i in xrange(batchSize):
            yield unlabledInstance( midiread(self.fpaths[self.index], self.roll,
                                             self.speriod).piano_roll.astype(theano.config.floatX) )
            self.index = self.index + 1

    def has_nextInstance(self, size):
        return self.index < len(self.fpaths)

    def read_all(self):
        '''
        Read all the instances availble
        '''
        dataset = [unlabledInstance( midiread(f, self.roll,
                            self.speriod).piano_roll.astype(theano.config.floatX) )
                   for f in self.fpaths]
        return dataset

class midiLabledDataReader(midiUnlabledDataReader):
    def __init__(self, num_of_targets, dataPath, roll=(21, 109), speriod=0.3):
        midiUnlabledDataReader.__init__(self, dataPath, roll, speriod)
        self.num_of_targets = num_of_targets

    def get_numOf_Targets(self):
        return self.num_of_targets

    def read_instance(self, batchSize):
        for i in xrange(batchSize):
            yield instance( midiread(self.fpaths[self.index], self.roll,
                                             self.speriod).piano_roll.astype(theano.config.floatX),
                            os.path.basename(self.fpaths[self.index]) )

    def read_all(self):
        dataset = [instance( midiread(f, self.roll,
                                              self.speriod).piano_roll.astype(theano.config.floatX),
                             os.path.basename(self.fpaths[self.index]) )
                   for f in self.fpaths]
        return  dataset
