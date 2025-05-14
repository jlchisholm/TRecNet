import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, concatenate, Masking, LSTM, TimeDistributed, Lambda, Reshape, Multiply, BatchNormalization, Bidirectional
from tensorflow.keras import regularizers 
from tensorflow.keras import initializers
import h5py 
import os 
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K  
from tensorflow.keras.optimizers import * 
import glob
import re
import importlib 
os.environ["CUDA_VISIBLE_DEVICES"]="1"

# Required python scripts and files 
X_maxmean = np.load('X_maxmean1.npy')
Y_maxmean = np.load('Y_maxmean1.npy')
# normalize.py, shape_timesteps.py, utilities.py needed as well 

# Parameters ============================================================================
# Path to h5 file
name = './../../../../../data/hongtao/inputs_2021-02-05/variables_zprime2750_rmu.h5'
# prediction save file
pred_name = 'preds.npy'
# true save file
true_name = 'true.npy'
# Predictions are saved in a (#samples)x20 numpy array, with 
# ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m', 'tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']
# ordering
# =======================================================================================

name0 = os.path.basename(name)
name0 = re.sub('\.h5$', '', name0)
dataset = h5py.File(name,'r')
size = np.array(dataset.get('th_pt')).size
X_keys = ['j1_pt', 'j1_eta', 'j1_phi', 'j1_m', 'j1_DL1r', 'j2_pt', 'j2_eta', 'j2_phi', 'j2_m', 'j2_DL1r', 'j3_pt', 'j3_eta', 'j3_phi', 'j3_m', 'j3_DL1r', 'j4_pt', 'j4_eta', 'j4_phi', 'j4_m', 'j4_DL1r', 'j5_pt', 'j5_eta', 'j5_phi', 'j5_m', 'j5_DL1r', 'j6_pt', 'j6_eta', 'j6_phi', 'j6_m', 'j6_DL1r', 'lep_pt', 'lep_eta', 'lep_phi', 'met_met', 'met_phi']
Y_keys = ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m', 'tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']

phi_keys = list(filter(lambda a: 'phi' in a, dataset.keys()))
eta_keys = list(filter(lambda a: 'eta' in a, dataset.keys()))
pt_keys =  list(filter(lambda a: 'pt' in a, dataset.keys()))
m_keys = list(filter(lambda a: 'm' in a, dataset.keys()))
DL1r_keys = list(filter(lambda a: 'DL1r' in a, dataset.keys()))

Y_length = len(Y_keys)
X_length = len(X_keys)
crop0 =  size 

import normalize
import shape_timesteps

Scaler = normalize.Scale_variables()
X_total, X_names = Scaler.scale_arrays(X_keys, X_maxmean)
Y_total, Y_names = Scaler.scale_arrays(Y_keys, Y_maxmean)

# print('Max scaling error: {}'.format(error))

split = int(np.floor(0.95*crop0)) 
split = 0 

trainY, testY = Y_total[0:split,:], Y_total[split:,:]

timestep_builder = shape_timesteps.Shape_timesteps()

totalX_jets, totalX_other = timestep_builder.reshape_X(X_total, X_names, False,True)

trainX_jets, testX_jets = totalX_jets[0:split,:,:], totalX_jets[split:,:,:]
trainX_other, testX_other = totalX_other[0:split,:], totalX_other[split:,:]

if testX_jets.shape[0] > 0: 
    other_bins = np.linspace(-1, 1, 40)
    phi_bins = np.linspace(-1, 1, 40)
    pt_bins = np.linspace(-1, 1, 40)
    Y_bins = [phi_bins if 'phi' in name else pt_bins if 'pt' in name else other_bins for name in Y_names]

    model = keras.models.load_model('./../runs/12-3-clean.keras')

    predictions_unscaled = model.predict([testX_jets, testX_other])
    true_unscaled = testY 

    total_predictions = model.predict([np.append(trainX_jets,testX_jets,axis=0), np.append(trainX_other,testX_other,axis=0)])
    Y_total, _ = Scaler.scale_arrays(Y_keys, Y_maxmean)

    predictions_origscale = Scaler.invscale_arrays(total_predictions, Y_names, Y_maxmean)[split:,:]
    true_origscale = Scaler.invscale_arrays(Y_total, Y_names, Y_maxmean)[split:,:]

    np.save(pred_name, predictions_origscale)
    np.save(true_name, true_origscale)
else:
    print('skip')
dataset.close()