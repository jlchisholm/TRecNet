#########################################################################
#                                                                       #
#  machine_learning.py                                                  #
#  Author: Jenna Chisholm                                               #
#  Updated: May.8/23                                                    #
#                                                                       #
#  Defines classes and functions relevant for testing neural networks.  # 
#                                                                       #
#  Thoughts for improvements: Have X and Y keys as input variables?     #
#                                                                       #
#########################################################################


import os, sys, time
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")
from argparse import ArgumentParser

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
#from clr_callback import * 

from MLUtil import *
from Scaler import Scaler



class Testing:

    def __init__(self, test_file, xmm_file, ymm_file, data_type):
        self.test_file = test_file
        self.xmm_file = xmm_file
        self.ymm_file = ymm_file
        self.data_type = data_type
        #self.onlyX = False if data_type=='nominal' else True
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None

    def save_results(self, model_id, preds_df, truths_df, save_loc):

        print('Saving results ...')

        save_name = save_loc+model_id+'_'+self.data_type+'_Results.root'
        results_file = uproot.recreate(save_name)

        if self.data_type=='nominal': 
            results_file["reco"] = {key:preds_df[key] for key in list(preds_df.keys())}
            results_file["parton"] = {key:truths_df[key] for key in list(truths_df.keys())}
        else:
            results_file[self.data_type] = {key:preds_df[key] for key in list(preds_df.keys())}

        print('Results saved in %s.' % save_name)




    def test(self, model_id, save_loc,):
        """
        Runs a trained model on a test dataset and saves the results.

            Parameters:
                model_id (str): ID of the trained model.
                save_loc (str): Location to save the test results.
        """
        
        # Grab important model info from the model id
        model_v = model_id.split('v')[0] + 'v' + model_id.split('v')[1].split('_')[0]
        n_jets = model_id.split('jets')[0].split('_')[-1]
        add_ttbar = True if '+ttbar' in model_id else False
        extra_b_mode = 'bbbar' if 'bbbar' in model_id else 'b1b2' if 'b1b2' in model_id else None

        # Create an object to use utilities
        processor = Utilities()
        
        # Load the things we'll need
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)
        trained_model = keras.models.load_model('trained_models/'+model_id+'/'+model_id+'.keras')

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in) OR read them in from the info file
        self.X_keys, self.Y_keys = processor.getInputKeys(model_v,n_jets,add_ttbar,extra_b_mode)

        # Pre-process the data
        testX_jets, testX_other, _, self.X_scaled_keys, self.Y_scaled_keys = processor.scale_and_shape(self.test_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, n_jets, -2)  # Mask value hard coded to -2

        # Predictions and truth BEFORE they're back to the original scale
        preds_scaled = trained_model.predict([testX_jets, testX_other])

        # Invert scaling
        scaler = Scaler()
        preds_origscale_dic = scaler.invscale_arrays(preds_scaled, self.Y_scaled_keys, Y_maxmean_dic)

        # Get the true values, if available
        if self.data_type=='nominal':
            with h5py.File(self.test_file,'r') as dataset:
                truth_keys = list(filter(lambda a: 'th_' in a or 'tl_' in a or 'ttbar_' in a or 'wh_' in a or 'wl_' in a, dataset.keys()))
                if extra_b_mode=='bbbar':
                    truth_keys.extend(list(filter(lambda a: 'b_' in a or 'bbar_' in a, dataset.keys())))
                elif extra_b_mode=='b1b2':
                    truth_keys.extend(list(filter(lambda a: 'b1_' in a or 'b2_' in a, dataset.keys())))
                truths_df = pd.DataFrame({key:dataset.get(key) for key in truth_keys})
        else:
            truths_df=None
                
        # Calculate all the same variables we had for truth, using the predicted values
        preds_df = pd.DataFrame(preds_origscale_dic)

        # Save results
        self.save_results(model_id, preds_df, truths_df, save_loc)