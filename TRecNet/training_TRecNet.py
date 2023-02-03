# Import some paths for where to look for classes
import sys
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")

# Import basic useful python packages
import numpy as np
import matplotlib.pyplot as plt
import h5py 

# Import basic useful packages for machine learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv1D, Flatten, Dense, Input, concatenate, Masking, LSTM, TimeDistributed, Lambda, Reshape, Multiply, BatchNormalization, Bidirectional
from tensorflow.keras import regularizers 
from tensorflow.keras import initializers
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
import tensorflow.keras.backend as K  
from tensorflow.keras.optimizers import * 

# Not sure we need these?

#import importlib
#import os 
#from clr_callback import *

# Import useful helper classes we've made
#from training_utilities import PlotLearning
import normalize
import shape_timesteps
#import analysis





# --- LOADING FILES --- #

print('Loading data ...')

# Data type we want to use
data_tag = 'variables_ttbar_ljets_6j_train'

# Dataset to train and test on
dataset = h5py.File('/mnt/xrootdg/jchishol/mntuples_08_01_22/'+data_tag+'.h5','r')

# Numpy dictionary of [max,mean] for each variable
X_maxmean_dic = np.load('/home/jchishol/TRecNet/X_maxmean_'+data_tag+'.npy',allow_pickle=True).item()
Y_maxmean_dic = np.load('/home/jchishol/TRecNet/Y_maxmean_'+data_tag+'.npy',allow_pickle=True).item()



# --- USEFUL VARIABLE DEFINITIONS --- #

# X and Y variables to be used
X_keys = ['j1_pt', 'j1_eta', 'j1_phi', 'j1_m', 'j1_isbtag', 'j2_pt', 'j2_eta', 'j2_phi', 'j2_m', 'j2_isbtag', 'j3_pt', 'j3_eta', 'j3_phi', 'j3_m', 'j3_isbtag', 'j4_pt', 'j4_eta', 'j4_phi', 'j4_m', 'j4_isbtag', 'j5_pt', 'j5_eta', 'j5_phi', 'j5_m', 'j5_isbtag', 'j6_pt', 'j6_eta', 'j6_phi', 'j6_m', 'j6_isbtag', 'lep_pt', 'lep_eta', 'lep_phi', 'met_met', 'met_phi']
Y_keys = ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m', 'tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']
Y_length = len(Y_keys)
X_length = len(X_keys)

# Variables keys, needed for helper codes
phi_keys = list(filter(lambda a: 'phi' in a, dataset.keys()))
eta_keys = list(filter(lambda a: 'eta' in a, dataset.keys()))
pt_keys =  list(filter(lambda a: 'pt' in a, dataset.keys()))
m_keys = list(filter(lambda a: 'm' in a, dataset.keys()))
isbtag_keys = list(filter(lambda a: 'isbtag' in a, dataset.keys()))

# Number of events in the dataset
size = np.array(dataset.get('th_pt')).size
crop0 = size    # Used in helper codes

# Get the maxmean values we actually need for this particular model
xmm_keys = list(X_maxmean_dic.keys())
ymm_keys = [key for key in Y_maxmean_dic.keys() if 'isTruth' not in key and 'ttbar' not in key]   # Don't want jet match tags or ttbar variables for this model
X_maxmean = np.array([X_maxmean_dic[key] for key in xmm_keys])
Y_maxmean = np.array([Y_maxmean_dic[key] for key in ymm_keys])



# --- PRE-PROCESS THE DATA --- #

# Scales data set to be between -1 and 1, with a mean of 0
#importlib.reload(normalize)
print('Pre-processing variables ...')
Scaler = normalize.Scale_variables(phi_keys,dataset,crop0)
X_total, X_names = Scaler.scale_arrays(dataset,X_keys,X_maxmean,crop0)
Y_total, Y_names = Scaler.scale_arrays(dataset,Y_keys, Y_maxmean,crop0)



# --- SPLIT DATA INTO TRAINING AND VALIDATION --- #

print('Splitting data ...')

# Set how the data will be split (70 for training, 15 for validation, 15 for testing)
# Taking ~82% for training and ~18% for validation amounts to ~70% of total being training and ~15% of total being validation
split = 70/85   # Gave 85% to train file, now want 70% for the actual training

# Split up jets and other for X
timestep_builder = shape_timesteps.Shape_timesteps()
totalX_jets, totalX_other = timestep_builder.reshape_X(dataset,X_total, X_names, phi_keys,crop0,False,True)

# Split into training and validation data
trainX_jets, valX_jets, trainX_other, valX_other, trainY, valY = train_test_split(totalX_jets, totalX_other, Y_total, train_size=split)



# --- BUILD THE MODEL --- #

def build_model():
    jet_input = Input(shape=(trainX_jets.shape[1], trainX_jets.shape[2]))
    Mask = Masking(-2)(jet_input)
    Maskshape = Reshape((trainX_jets.shape[1], trainX_jets.shape[2]))(Mask)
    other_input = Input(shape=(trainX_other.shape[1]))
    flat_jets =  Flatten()(jet_input)
    concat0 = concatenate([other_input, flat_jets])
    PreDense1 = Dense(256, activation='relu')(concat0)
    PreDense2 = Dense(256, activation='relu')(PreDense1)
    PreDense3 = Dense(trainX_jets.shape[1], activation='sigmoid')(PreDense2)
    Shape_Dot = Reshape((-1,1))(PreDense3)
    
    TDDense11 = TimeDistributed(Dense(128, activation='relu'))(Maskshape)
    TDDense12 = TimeDistributed(Dense(64, activation='relu'))(TDDense11)
    Dot_jets = Multiply()([Shape_Dot, TDDense12])
    TDDense13 = TimeDistributed(Dense(256, activation='relu'))(Dot_jets)
    TDDense14= TimeDistributed(Dense(256, activation='relu'))(TDDense13)
    flat_right = Flatten()(TDDense14)
    
    Dense21 = Dense(128, activation='relu')(other_input)
    Dense22 = Dense(64, activation='relu')(Dense21)
    flat_other = Flatten()(Dense22)
    
    concat = concatenate([flat_other, flat_right])
    
    ldense1 = Dense(256, activation='relu')(concat)
    ldense2 = Dense(128, activation='relu')(ldense1)
    loutput = Dense(len(Y_names)//2)(ldense2)
    
    hconcat = concatenate([loutput, concat])
    hdense1 = Dense(256, activation='relu')(hconcat)
    hdense2 = Dense(128, activation='relu')(hdense1)
    houtput = Dense(len(Y_names)//2)(hdense2)
    
    output = concatenate([houtput, loutput])
    
    model = keras.models.Model(inputs=[jet_input, other_input], outputs=output)
    

    # lr_schedule = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3, decay_steps=5000,decay_rate=0.6)
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=1e-3, decay_steps=10000,end_learning_rate=5e-5,power=0.25)
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(loss='mae', optimizer= optimizer, metrics=['mse'])
    
    return model 


# Build model
print('Building model ...')
model = build_model()
print(model.summary())
#keras.utils.plot_model(model,to_file='Model_Custom.png',show_shapes=True,show_dtype=True,show_layer_names=True)




# --- TRAIN THE MODEL AND SAVE IT --- #

print('Training model ...')

# Set when to stop (has to do with loss?)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=4)  # patience=4 means stop training after 4 epochs with no improvement

# Set number of epochs (runs through?)
Epochs= 256


# Fit the model (i.e. TRAIN the model)
history = model.fit([trainX_jets, trainX_other], trainY, verbose=1, epochs=Epochs,
                   validation_data=([valX_jets, valX_other], valY), shuffle=True, callbacks=[early_stop],
                    batch_size=1000)
#history = model.fit([totalX_jets, totalX_other], Y_total, verbose=1, epochs=Epochs,
#                   validation_split=0.1, shuffle=True, callbacks=[early_stop],
#                    batch_size=1000)

print('Model training complete. Saving model and training history ...')

model.save('Model_Custom_full_6j.keras')
np.save('trainHistory_Model_Custom_full_6j.npy',history.history)



# --- SAVE LOSS FUNCTION --- #

plt.figure(figsize=(9,6))
plt.plot(history['loss'], label='training')
plt.plot(history['val_loss'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('MAE Loss')
plt.legend()
plt.title('TRecNet MAE Loss')
plt.savefig('~/TRecNet/TRecNet/TRecNet_MAE_Loss',bbox_inches='tight')
print('Saved: ~/TRecNet/TRecNet/TRecNet_MAE_Loss.png')

plt.figure(figsize=(9,6))
plt.plot(history['mse'], label='training')
plt.plot(history['val_mse'], label='validation')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('~/TRecNet/TRecNet/TRecNet_MSE_Loss',bbox_inches='tight')
print('Saved: ~/TRecNet/TRecNet/TRecNet_MSE_Loss.png')




