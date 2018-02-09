import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
RUNALL = False

"""
Read in train and test as Pandas DataFrames
"""
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

# randomly sample a portion of df_train
# df_train = df_train.sample(n=10000)

#store gap values
Y_train = df_train.gap.values
#row where testing examples start
test_idx = df_train.shape[0]
#delete 'Id' column
df_test = df_test.drop(['Id'], axis=1)
#delete 'gap' column
df_train = df_train.drop(['gap'], axis=1)

#DataFrame with all train and test examples so we can more easily apply feature engineering on
df_all = pd.concat((df_train, df_test), axis=0)
df_all.head()

# adding features
from __future__ import print_function
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors

# Feature engineering
# store smiles
smiles = df_all.smiles

start = time.time()

carbons = []
carbons = np.vstack(df_all.smiles.apply(lambda x: x.count('c')))
df_all['carbons'] = carbons

end = time.time()
print("how long", (end - start))

double = []
double = np.vstack(df_all.smiles.apply(lambda x: x.count('=')))
df_all['double'] = double

end = time.time()
print("how long", (end - start))

single = []
single = np.vstack(df_all.smiles.apply(lambda x: x.count('-')))
df_all['single'] = single

end = time.time()
print("how long", (end - start))

nitrogen = []
nitrogen = np.vstack(df_all.smiles.apply(lambda x: x.count('n')))
df_all['nitrogen'] = nitrogen

end = time.time()
print("how long", (end - start))

oxygen = []
oxygen = np.vstack(df_all.smiles.apply(lambda x: x.count('o')))
df_all['oxygen'] = oxygen

end = time.time()
print("how long", (end - start))

ccccc = []
ccccc = np.vstack(df_all.smiles.apply(lambda x: x.count('ccccc')))
df_all['ccccc'] = ccccc

end = time.time()
print("how long", (end - start))

ccc = []
ccc = np.vstack(df_all.smiles.apply(lambda x: x.count('ccc')))
df_all['ccc'] = ccc

end = time.time()
print("how long", (end - start))

Si = []
Si = np.vstack(df_all.smiles.apply(lambda x: x.count('Si')))
df_all['Si'] = Si

end = time.time()
print("how long", (end - start))

se = []
se = np.vstack(df_all.smiles.astype(str).apply(lambda x: x.count('se')))
df_all['se'] = se

end = time.time()
print("how long", (end - start))

mols = df_all.smiles.astype(str).apply(lambda x: Chem.MolFromSmiles(x))
df_all['mols'] = mols

atoms = np.vstack(df_all.mols.apply(lambda x: x.GetNumAtoms()))
df_all['atoms'] = atoms
bonds = np.vstack(df_all.mols.apply(lambda x: x.GetNumBonds()))
df_all['bonds'] = bonds

# aromaticity
aro = []
aro = np.vstack(df_all.mols.apply(lambda x: sum(int(x.GetAtomWithIdx(i).GetIsAromatic()) for i in range(x.GetNumAtoms()))))
df_all['aro'] = aro

# adding new features

# TPSA BAAAAAD
# tpsa = np.vstack(df_all.smiles.apply(lambda x: Descriptors.TPSA(Chem.MolFromSmiles(x))))
# df_all['tpsa'] = pd.DataFrame(tpsa)

# sp3 hybridization
sp3 = np.vstack(df_all.mols.apply(lambda x: rdMolDescriptors.CalcFractionCSP3(x)))
df_all['sp3'] = sp3


#Drop the 'smiles' column
df_all = df_all.drop(['smiles'], axis=1)
df_all = df_all.drop(['mols'], axis=1)
vals = df_all.values
X_train = vals[:test_idx]
X_test = vals[test_idx:]
print("Train features:", X_train.shape)
print("Train gap:", Y_train.shape)
print("Test features:", X_test.shape)

hd = np.vstack(mols.apply(lambda x: Descriptors.NumHDonors(x)))
df_all['hd'] = hd

ha = np.vstack(mols.apply(lambda x: Descriptors.NumHAcceptors(x)))
df_all['ha'] = ha

aliphcarbo = np.vstack(mols.apply(lambda x: Descriptors.NumAliphaticCarbocycles(x)))
df_all['aliphcarbo'] = aliphcarbo

aliphhetero = np.vstack(mols.apply(lambda x: Descriptors.NumAliphaticHeterocycles(x)))
df_all['aliphhetero'] = aliphhetero

aliphrings = np.vstack(mols.apply(lambda x: Descriptors.NumAliphaticRings(x)))
df_all['aliphrings'] = aliphrings

arocarbos = np.vstack(mols.apply(lambda x: Descriptors.NumAromaticCarbocycles(x)))
df_all['arocarbos'] = arocarbos

aroheteros = np.vstack(mols.apply(lambda x: Descriptors.NumAromaticHeterocycles(x)))
df_all['aroheteros'] = aroheteros

arorings = np.vstack(mols.apply(lambda x: Descriptors.NumAromaticRings(x)))
df_all['arorings'] = arorings

heteros = np.vstack(mols.apply(lambda x: Descriptors.NumHeteroatoms(x)))
df_all['heteros'] = heteros

radelecs = np.vstack(mols.apply(lambda x: Descriptors.NumRadicalElectrons(x)))
df_all['radelecs'] = radelecs

rotbonds = np.vstack(mols.apply(lambda x: Descriptors.NumRotatableBonds(x)))
df_all['rotbonds'] = rotbonds

satcarbos = np.vstack(mols.apply(lambda x: Descriptors.NumSaturatedCarbocycles(x)))
df_all['satcarbos'] = satcarbos

satheteros = np.vstack(mols.apply(lambda x: Descriptors.NumSaturatedHeterocycles(x)))
df_all['satheteros'] = satheteros

satrings = np.vstack(mols.apply(lambda x: Descriptors.NumSaturatedRings(x)))
df_all['satrings'] = satrings

valelecs = np.vstack(mols.apply(lambda x: Descriptors.NumValenceElectrons(x)))
df_all['valelecs'] = valelecs

from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# Models

# fn that returns trained linear function to make predictions
def train_and_test_linear_model(X_train, Y_train):
    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    # return the prediction we've trained
    return LR.predict



# RIDGE REG
def train_and_test_ridge_model(alpha):
    def helper(X_train, Y_train):
        reg = linear_model.Ridge(alpha=alpha)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper


# LASSO REG
def train_and_test_lasso_model(alpha):
    def helper(X_train, Y_train):
        reg = linear_model.Lasso(alpha=alpha)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper


# ELASTICNET REG
def train_and_test_elastic_net_model(alpha):
    def helper(X_train, Y_train):
        reg = linear_model.ElasticNet(alpha=alpha)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper

# SVM REG
def train_and_test_svm_model():
    def helper(X_train, Y_train):
        reg = SVR(kernel='rbf', C=1.0, epsilon=0.2)
        reg.fit(X_train, Y_train)
        return reg.predict
    return helper

# RF pred
# fn that returns trained Random Forest to make predictions
def train_and_test_RF_model(X_train, Y_train):
    RF = RandomForestRegressor(n_estimators=20)
    RF.fit(X_train, Y_train)
    # return the prediction we've trained
    return RF.predict

from training_and_validation import select_best_model

if RUNALL:
    select_best_model(X_train, Y_train, 4, train_and_test_linear_model)
    
if RUNALL:
    select_best_model(X_train, Y_train, 2, train_and_test_ridge_model(0.1))

if RUNALL:
    select_best_model(X_train, Y_train, 4, train_and_test_lasso_model(0.1))
    
if RUNALL:
    select_best_model(X_train, Y_train, 4, train_and_test_elastic_net_model(0.1))
    
if RUNALL:
    select_best_model(X_train, Y_train, 4, train_and_test_svm_model())
    
if RUNALL:
    select_best_model(X_train, Y_train, 2, train_and_test_RF_model)
    
def write_to_file(filename, predictions):
    with open(filename, "w") as f:
        f.write("Id,Prediction\n")
        for i,p in enumerate(predictions):
            f.write(str(i+1) + "," + str(p) + "\n")
  
# write a file for ridge regression
rid_reg = linear_model.Ridge(alpha=0.1)
rid_reg.fit(X_train, Y_train)
rid_reg_pred = rid_reg.predict(X_test)
write_to_file("rid_reg_correct_more_features.csv", rid_reg_pred)
print("starting RF")
# write a file for RF
rf = RandomForestRegressor(n_estimators=20)
rf.fit(X_train, Y_train)
rf_pred = rf.predict(X_test)
write_to_file("rf_correct_more_features.csv", rf_pred)

from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

X_red = PCA_red(X_train)

estimators = [('k_means_15', KMeans(n_clusters=15)),
              ('k_means_iris_10', KMeans(n_clusters=10)),
              ('k_means_iris_5', KMeans(n_clusters=5))]

fignum = 1
titles = ['15 clusters', '10 clusters', '5 clusters']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(8, 6))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X_red)
    labels = est.labels_
    ax.scatter(X_red[:, 1], X_red[:, 0], X_red[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('PC2')
    ax.set_ylabel('PC1')
    ax.set_zlabel('PC3')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1


# Plot the ground truth
fig = plt.figure(fignum, figsize=(8, 6))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

#labels = Y_train*10-10
#y = labels.astype(np.float)
    
ax.scatter(X_red[:, 1], X_red[:, 0], X_red[:, 2], c=Y_train, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('PCA 2')
ax.set_ylabel('PCA 1')
ax.set_zlabel('PCA 3')
ax.set_title('Real Y Train')
ax.dist = 12

plt.show()

# PCA theory and learning:
from sklearn.preprocessing import StandardScaler

# standardize data to mean=0, var=1
X_std = StandardScaler().fit_transform(X_train)

# compute covariance matrix
cov_mat = np.cov(X_std.T)

#eigendecomposition - make real
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_vals = np.real(eig_vals)
eig_vecs = np.real(eig_vecs)

# Singular Vector Decomposition 
u, s, v = np.linalg.svd(X_std.T)

# select principal components
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key=lambda x: x[0])
eig_pairs.reverse()
eig_vals_sorted = [x[0] for x in eig_pairs]

idx = [i for i in range(len(eig_pairs)) ]

plt.figure()
plt.scatter(idx, eig_vals_sorted)
plt.show()

import numpy as np
from sklearn.model_selection import KFold

# Validation

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
    return error / float(num_validation_sets)