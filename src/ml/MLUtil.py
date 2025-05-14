import os, sys, time
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training

import numpy as np
import vector
import itertools
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py
import uproot

import tensorflow as tf
from tensorflow import keras
from keras.layers import Conv1D, Flatten, Dense, Input, concatenate, Masking, LSTM, TimeDistributed, Lambda, Reshape, Multiply, BatchNormalization, Bidirectional
from keras import regularizers 
from keras import initializers
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard
import keras.backend as K  
from keras.optimizers import *
import keras_tuner as kt
#from clr_callback import * 

import TRecNet.src.ml.normalize_new as normalize_new
import TRecNet.src.ml.shape_timesteps_new as shape_timesteps_new



class TRecNet_Model:
    """
    A class for creating a machine learning model object.
    """

    def __init__(self, model_name, model_id=None, n_jets=None):
        """
        Initializes a machine learning model object.

            Parameters:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                model_id (str): Unique model identifier (default: None).
                n_jets (int): Number of jets the model is trained with (default: None).

            Attributes:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                model_id (str): Unique model identifier, based on model name, number of jets, and time it was created.
                mask_value
                n_jets (int): Number of jets the model is trained with.
                model (keras.model): Trained keras model.
                training_time
                training_history (keras.model.history.history): Training history for the model.
        """
        
        self.model_name = model_name
        self.model_id = time.strftime(model_name+"_"+str(n_jets)+"jets_%Y%m%d_%H%M%S") if model_id==None else model_id   # Model unique save name (based on the date)
        self.n_jets = n_jets if model_id==None else int(model_id.split('_')[1].split('jets')[0]) # If not given, get from model_id
        self.mask_value = -2   # Define here so it's consist between model building and jet timestep building
        self.model = None
        self.frozen_model_id = None
        self.training_time = None
        self.training_history = None



class Utilities:
    """
    A class containing useful functions for training, validating, and testing.
    
        Methods:
            getInputKeys: Gets lists of the (original scale) X and Y variable keys.
            loadMaxMean: Loads the max-mean dictionaries.
            scale: Scale the X and Y data such that they have a mean of 0, range of (-1,1), and encode phi variables in px and py (or cosine and sine).
            unscale: Unscale the X and Y data back to the original scale and variables.
            prepData: Prepares data for training by performing a mean-max scaling and phi-encoding, and then splitting dataset into (time-stepped) jets, other, and y data.
    """

    def __init__(self):
        pass

    def getInputKeys(self, model_name, n_jets):
        """
        Gets lists of the (original scale) X and Y variable keys.

            Parameters:
                model_name (str): Name of the model (e.g. 'TRecNet+ttbar').
                n_jets (int): Number of jets the model is or will be trained on.
                
            Returns:
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
        """

        X_keys = ['j'+str(i+1)+'_'+v for i, v in itertools.product(range(n_jets),['pt','eta','phi','m','isbtag'])] + ['lep_pt', 'lep_eta', 'lep_phi', 'met_met', 'met_phi']
        Y_keys = ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m', 'tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']
        if 'ttbar' in model_name:
            Y_keys.extend(['ttbar_pt','ttbar_eta','ttbar_phi','ttbar_m'])
        if model_name=='JetPretrainer': 
            Y_keys = ['j'+str(i+1)+'_isTruth' for i in range(n_jets)]

        return X_keys, Y_keys


    def loadMaxMean(self, xmm_file, ymm_file):
        """
        Loads the max-mean dictionaries.
        
            Parameters:
                xmm_file (str): X maxmean file name (including path).
                ymm_file (str): Y maxmean file name (including path).

            Returns:
                X_maxmean_dic (dict): Dictionary of max and mean for X variables.
                Y_maxmean_dic (dict): Dictionary of max and mean for Y variables.
        """

        X_maxmean_dic = np.load(xmm_file,allow_pickle=True).item()
        Y_maxmean_dic = np.load(ymm_file,allow_pickle=True).item()

        return X_maxmean_dic, Y_maxmean_dic


    def scale(self, dataset, X_keys, Y_keys, X_maxmean_dic, Y_maxmean_dic, onlyX=False):
        """
        Scale the X and Y data such that they have a mean of 0, range of (-1,1), and encode phi variables in px and py (or cosine and sine).

            Parameters:
                dataset (h5py dataset): Training data.
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
                X_maxmean_dic (dict): Dictionary of max and mean for X variables.
                Y_maxmean_dic (dict): Dictionary of max and mean for Y variables.
                
            Optional:
                onlyX (bool): Whether or not to also prepare Y Data, or only prepare X data (default: False).

            Returns:
                X_df (pd.DataFrame): Scaled X data.
                Y_df (pd.DataFrame): Scaled Y data.
                scaled_X_keys (list of str): Scaled X keys.
                scaled_Y_keys (list of str): Scaled Y keys.
        """        

        scaler = normalize_new.Scaler()
        X_df = scaler.scale_arrays(dataset, X_keys, X_maxmean_dic)
        scaled_X_keys = X_df.keys()
        
        if onlyX:
            Y_df = None
            scaled_Y_keys = scaler.get_scaled_Ykeys(Y_keys)
        else:
            Y_df = scaler.scale_arrays(dataset, Y_keys, Y_maxmean_dic)
            scaled_Y_keys = Y_df.keys()

        return X_df, Y_df, scaled_X_keys, scaled_Y_keys



    def prepData(self, datafile, X_maxmean_dic, Y_maxmean_dic, X_keys, Y_keys, jn, mask_value, onlyX=False):
        """
        Prepares data for training by performing a mean-max scaling and phi-encoding, and then splitting dataset into (time-stepped) jets, other, and y data.

            Parameters:
                datafile (str): File name (and path) for training dataset.
                xmm_file (str): Path and file name for the X_maxmean file to be used in scaling.
                X_keys (list of str): Names of the (original) input variables.
                ymm_file (str): Path and file name for the Y_maxmean file to be used in scaling.
                Y_keys (list of str): Names of the (original) output variables.
                jn (int): Number of jets we're training with.
                mask_value (int): Value to mask non-existent jets with.
                
            Optional:
                onlyX (bool): Whether or not to also prepare Y Data, or only prepare X data (default: False).

            Returns:
                totalX_jets (np.array): Scaled, time-stepped jets.
                totalX_other (np.array): Other scaled input data.
                Y_total (np.array): Scaled output data.
                scaled_X_keys (list of str): Names of the (scaled) input variables.
                scaled_Y_keys (list of str): Names of the (scaled) output variables.
        """

    
        print('Preparing data...')
        
        with h5py.File(datafile,'r') as dataset:   # Only want the dataset open as long as we need it
            
            # Create the timestep builder while we still have the dataset open
            timestep_builder = shape_timesteps_new.Shape_timesteps(dataset, jn, mask_value)
            
            # Scales data set to be between -1 and 1, with a mean of 0, and encodes phi in other variables (e.g. px, py)
            X_df, Y_df, scaled_X_keys, scaled_Y_keys = self.scale(dataset, X_keys, Y_keys, X_maxmean_dic, Y_maxmean_dic, onlyX)

        # Split up jets and other for X, and Y just all stays together
        totalX_jets, totalX_other = timestep_builder.reshape_X(X_df)
        Y_total = np.array(Y_df)
        
        return totalX_jets, totalX_other, Y_total, scaled_X_keys, scaled_Y_keys
