"""
Extract MFCC features and perform Mean Normalization.
"""
import numpy as np

import base


def mfcc_features(sig):
    features = base.mfcc(sig, samplerate=44100, winlen=0.02, winstep=0.01,
                         numcep=13, nfilt=40)

    # Mean Normalization for feature vectors.
    mean_vector = np.mean(features, axis=0)
    normalized = features - mean_vector
    return normalized

