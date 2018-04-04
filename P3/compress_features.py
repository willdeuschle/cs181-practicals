import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import librosa

df_train = pd.read_csv("train.csv.gz", chunksize=200, header=None)

CLASSES = np.array([i for i in range(10)])
SAMPLE_RATE = 22050
# FOURIER TRANSFORM PARAMETERS
BINS_OCTAVE = 12*2
N_OCTAVES = 7
NUM_BINS = BINS_OCTAVE * N_OCTAVES

def extract_features(data_pt):
    # short time fourier, as we were doing before
    short_time_fourier = np.abs(librosa.stft(data_pt))
    # latice of tones avg
    tonnetz_measure = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(data_pt), sr=SAMPLE_RATE).T,axis=0)
    # mfcc, power spectrum
    mel_filter = np.mean(librosa.feature.mfcc(y=data_pt, sr=SAMPLE_RATE, n_mfcc=40).T,axis=0)
    # chromagram from power spectrum
    chroma_short_time = np.mean(librosa.feature.chroma_stft(S=short_time_fourier, sr=SAMPLE_RATE).T,axis=0)
    # mel spec they provided us with
    mel_spec = np.mean(librosa.feature.melspectrogram(data_pt, sr=SAMPLE_RATE).T,axis=0)
    # contrast recommended by '02 paper
    contrast = np.mean(librosa.feature.spectral_contrast(S=short_time_fourier, sr=SAMPLE_RATE).T,axis=0)
    # send it all back
    return tonnetz_measure, mel_filter, chroma_short_time, mel_spec, contrast

for idx, chunk in enumerate(df_train):
    print(f"current idx: {idx}")
    features = np.empty((0,194))
    train = np.array(chunk)
    N_train = train.shape[0]

    X_train = train[:,:-1]
    Y_train = train[:,-1]
    Y_train = Y_train.reshape(N_train)
    
    for idx, data_pt in enumerate(X_train):
        class_val = Y_train[idx]
        tonnetz_measure, mel_filter, chroma_short_time, mel_spec, contrast = extract_features(data_pt)
        features = np.vstack([features, np.hstack([tonnetz_measure, mel_filter, chroma_short_time, mel_spec, contrast, class_val])])
    with open('transformed_train.csv', 'a') as f:
        np.savetxt(f, features, delimiter=',')