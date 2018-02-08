# comment
import numpy as np
from sklearn.model_selection import KFold

# to compute RSM error
def compute_error(pred, actual):
    num_pts = float(pred.shape[0])
    # RSM error
    return np.sqrt(np.sum(np.square(np.subtract(pred, actual))) / num_pts)

# estimate model performance using validation sets
def select_best_model(X, Y, num_validation_sets, model_fn):
    # error accumulator
    error = 0
    # set up the kfold operation
    kf = KFold(n_splits=num_validation_sets)
    kf.get_n_splits(X)
    for train_index, validation_index in kf.split(X):
        print("next validation set")
        # get train and validate data
        X_train, X_validation = X[train_index], X[validation_index]
        Y_train, Y_validation = Y[train_index], Y[validation_index]
        # train, should return a function to make predictions
        print("training")
        trained_fn = model_fn(X_train, Y_train)
        # get error
        pred = trained_fn(X_validation)
        print("got a prediction!")
        error += compute_error(pred, Y_validation)
        
    final_error = error / float(num_validation_sets)
    print("Final error", final_error)
    return final_error
