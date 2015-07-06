import numpy as np

class gmotion(object):
    ''' read ground acceleration record
        eith 1-d numpy array or csv file'''

    def __init__(self, gmotion_file, dt_gmotion, period=None, psa=None):

        if isinstance(gmotion_file, np.ndarray):
            try:
                assert(gmotion_file.ndim==1)
                self.ground = gmotion_file
            except AssertionError:
                self.ground = gmotion_file.flatten()
        else:
            self.ground = np.loadtxt(gmotion_file)

        self.dt = dt_gmotion
        self.npts = len(self.ground)
        self.ts = np.linspace(0.0, (self.npts-1)*dt_gmotion, self.npts)

        self.period = period
        self.psa = psa
