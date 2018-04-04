# initial libs
import re
import numpy as np
import pandas as pd
import string
import matplotlib.pyplot as plt
import librosa

CLASS_WEIGHT = {
    0: .2784,
    1: .0236,
    2: .1129,
    3: .0778,
    4: .0942,
    5: .1099,
    6: .0021,
    7: .0903,
    8: .1002,
    9: .1107,
}

# incremental read of csv
CHUNK_SIZE = 200
df_train = pd.read_csv("train.csv.gz", chunksize=CHUNK_SIZE, header=None)

# constant information
CLASSES = np.array([i for i in range(10)])
SAMPLE_RATE = 22050
# FOURIER TRANSFORM PARAMETERS
BINS_OCTAVE = 12*2
N_OCTAVES = 7
NUM_BINS = BINS_OCTAVE * N_OCTAVES

# feature transformation function
# Given a wav time series, makes a mel spectrogram
# which is a short-time fourier transform with
# frequencies on the mel (log) scale.
def mel_spec(y):
    Q = librosa.cqt(y=y, sr=SAMPLE_RATE, bins_per_octave=BINS_OCTAVE,n_bins=NUM_BINS)
    Q_db = librosa.amplitude_to_db(Q,ref=np.max)
    return Q_db


# model
from sklearn.linear_model import SGDClassifier
# online training
def train_SGD_model(sgd_model, X_train, Y_train):
    sgd_model.partial_fit(X_train, Y_train, CLASSES)
    return

# instantiate our sgd_model
sgd_model = SGDClassifier(loss='log', class_weight=CLASS_WEIGHT)

def prep_chunk(chunk):
    # Just some re-shaping and dimension finding
    train = np.array(chunk)
    N_train = train.shape[0]

    X_train = train[:,:-1]
    Y_train = train[:,-1]
    Y_train = Y_train.reshape(N_train)

    # This means that the spectrograms are 168 rows (frequencies)
    # By 173 columns (time frames)
    test_spec = mel_spec(X_train[0])
    FEATS = test_spec.shape[0]
    FRAMES = test_spec.shape[1]

    tmp_train = np.zeros((N_train,FEATS,FRAMES))
    for i in range(N_train):
        tmp_train[i,:,:] = mel_spec(X_train[i])

    dims = np.shape(tmp_train)
    X_train = tmp_train.reshape(dims[0],dims[1]*dims[2])
    return (X_train, Y_train)
    

# now incrementally train
for idx, chunk in enumerate(df_train):
    if idx > 10:
        break
    print(f"chunk idx: {idx} / {6200 / CHUNK_SIZE}")
    
    # reshape data
    X_train, Y_train = prep_chunk(chunk)
    
    # now partially fit
    sgd_model.partial_fit(X_train, Y_train, CLASSES)
    
print("I'm done! test me")

def compute_perc_correct(pred, actual):
    num_pts = float(pred.shape[0])
    res = np.sum((pred - actual)==0)/num_pts
    return res

def prep_test_chunk(chunk):
    # Just some re-shaping and dimension finding
    test = np.array(chunk)
    N_test = test.shape[0]
    
    X_test = test[:,1:]
    sample_ids = test[:,0]
    sample_ids = sample_ids.reshape(N_test)

    # This means that the spectrograms are 168 rows (frequencies)
    # By 173 columns (time frames)
    test_spec = mel_spec(X_test[0])
    FEATS = test_spec.shape[0]
    FRAMES = test_spec.shape[1]

    tmp_test = np.zeros((N_test,FEATS,FRAMES))
    for i in range(N_test):
        tmp_test[i,:,:] = mel_spec(X_test[i])

    dims = np.shape(tmp_test)
    X_test = tmp_test.reshape(dims[0],dims[1]*dims[2])
    return (X_test, sample_ids)

# FOR GENERATING PREDICTIONS
# generate predictions on test set
# TEST_CHUNK_SIZE = 10
# df_test = pd.read_csv("test.csv.gz", chunksize=CHUNK_SIZE, header=None)
# for idx, chunk in enumerate(df_test):
#     print(f"chunk idx: {idx}")
    
#     # reshape data
#     X_test, test_ids = prep_test_chunk(chunk)
    
#     # predict
#     preds = sgd_model.predict(X_test)
#     print(f"Prediction: {preds}")
#     # now write these predictions and their ids to the csv
#     with open('sgd_submission.csv', 'a') as f:
#         for pred_idx, pred in enumerate(preds):
#             test_id = int(test_ids[pred_idx])
#             f.write(f"{test_id},{pred}\n")

# FOR TESTING
for i in range(10):
    testing_chunk = next(df_train)
    X_train, actual = prep_chunk(testing_chunk)
    pred = sgd_model.predict(X_train)
    print("prediction", pred)
    print("actual", actual)
    print(f"Percentage correct: {compute_perc_correct(pred, actual)}")