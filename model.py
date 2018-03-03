import os

from audio_gmm import AudioGMM
import mfcc
import gmm_identity


def scan_wave_files(path):
    files = []
    for file in os.listdir(path):
        if file.endswith(".wav"):
            files.append(file.strip('.wav'))
    return files


def validate_wave_labels(labels, path):
    valid_labels = []
    for lb in labels:
        f = os.path.join(path, lb + '.wav')
        if os.path.exists(f):
            valid_labels.append(lb)
    return valid_labels


class IdentificationModel:
    def __init__(self, labels=None, path='.'):
        self._validate_input(labels, path)
        self.candidates = len(self.labels)
        self._weights = [0] * self.candidates
        self._means = [0] * self.candidates
        self._covars = [0] * self.candidates

    def _validate_input(self, labels=None, path='.'):
        if labels is None:
            self.labels = scan_wave_files(path)
        else:
            self.labels = validate_wave_labels(labels, path)
        self.path = path

    def label2file(self, label):
        return os.path.join(self.path, label + '.wav')

    def train(self):
        for i in range(self.candidates):
            model = AudioGMM(32, self.label2file(self.labels[i]))
            model.initialize()
            self._weights[i] = model.weight()
            self._means[i] = model.mean()
            self._covars[i] = model.covar()

    def test(self, x):
        if self.candidates <= 0:
            return "unknown"

        x = mfcc.mfcc_features(x)
        idx = gmm_identity.get_identity(x, self.candidates, self._weights,
                                        self._means, self._covars)
        if 0 <= idx < self.candidates:
            return self.labels[idx]
        else:
            return "unknown"
