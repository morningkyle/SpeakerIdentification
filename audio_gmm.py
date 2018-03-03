from sklearn import mixture
import scipy.io.wavfile as wav

import mfcc 


class AudioGMM:
    def __init__(self, num, audio_file):
        self.num = num
        self.audio_file = audio_file
        self.mixture = None

    def initialize(self):
        (rate, sig) = wav.read(self.audio_file)
        features = mfcc.mfcc_features(sig)
        self.mixture = mixture.GaussianMixture(n_components=self.num, reg_covar=0.01, n_init=10).fit(features)

    def mean(self):
        if self.mixture:
            return self.mixture.means_
        else:
            return []

    def weight(self):
        if self.mixture:
            return self.mixture.weights_
        else:
            return []

    def covar(self):
        if self.mixture:
            return self.mixture.covariances_
        else:
            return []

