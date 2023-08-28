############################################################################
#                                                                          #
#  machine_learning.py                                                     #
#  Author: Jenna Chisholm                                                  #
#  Updated: May.8/23                                                       #
#                                                                          #
#  Defines classes and functions relevant for validating neural networks.  # 
#                                                                          #
#  Thoughts for improvements: Have X and Y keys as input variables?        #
#                                                                          #
############################################################################


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
from sklearn.metrics import multilabel_confusion_matrix
import seaborn as sb
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

ticks_size = 14
axis_size = 16
legend_size = 14




class Validating:
    
    def __init__(self, train_file, xmm_file, ymm_file):
        self.train_file = train_file
        self.xmm_file = xmm_file
        self.ymm_file = ymm_file
    
    def plot(self, model_name, var, preds, truths, save_loc):
        
        nbins = 30
        max_val = max(max(preds), max(truths))
        min_val = min(min(preds), min(truths))
        
        var_label = '$\phi^{t,had}$' if var=='th_phi' else var
        
        plt.figure()
        plt.hist(truths, bins=nbins, range=(min_val, max_val), histtype='step',label='Truth')
        plt.hist(preds, bins=nbins, range=(min_val, max_val), histtype='step',label='Prediction')
        if model_name == 'JetPretrainer': plt.hist(np.array(preds).round(), bins=nbins, range=(min_val, max_val), histtype='step',label='reco (rounded)')
        plt.legend(fontsize=legend_size, loc='lower right')
        plt.xlabel(var_label,fontsize=axis_size)
        plt.ylabel('Events',fontsize=axis_size)
        plt.xticks(fontsize=ticks_size)
        plt.yticks(fontsize=ticks_size)
        plt.savefig(save_loc+var, bbox_inches='tight')
        plt.close()
        
        
    def jet_cm_plot(self, preds, truths, save_loc):
        
        cms = multilabel_confusion_matrix(np.array(truths), np.array(preds).round())
        print(cms)
        #fig, axs = plt.subplots(2,3, sharex=True, sharey=True)

        fig, ax = plt.subplots(2,3, sharex=True, sharey=True, figsize=(10, 8))
        cbar_ax = fig.add_axes([1.02, .11, .03, .82])

        jet = 0
        for i in range(2):
            for j in range(3):
                cm = cms[jet]
                group_names = ['True Negative','False Positive','False Negative','True Positive']
                group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
                group_percentages = ["{0:.2%}".format(value) for value in cm[0].flatten()/np.sum(cm[0])]
                group_percentages.extend(["{0:.2%}".format(value) for value in cm[1].flatten()/np.sum(cm[1])])
                labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
                labels = np.asarray(labels).reshape(2,2)
                normed = [cm[0]*100/np.sum(cm[0]),cm[1]*100/np.sum(cm[1])]
                sub = sb.heatmap(normed,vmin=0, vmax=100, cmap='YlGnBu',fmt='',annot=labels, cbar=True, cbar_ax=cbar_ax,square=True, ax=ax[i,j])
                ax[i,j].set_title('Jet 1'+str(jet+1))
                ax[i,j].tick_params(left=False,bottom=False) 
                jet+=1
                #print('j'+str(jet+1)+' Accuracy: ',accuracy_score(true_unscaled[:,jet],predictions_unscaled.round()[:,jet]))
                #print('j'+str(jet+1)+' Precision: ',precision_score(true_unscaled[:,jet],predictions_unscaled.round()[:,jet]))


        fig.supxlabel('Reco')
        fig.supylabel('Truth')

        fig.tight_layout()
        fig.savefig(save_loc+"JetPretrain_Normalized_CM",bbox_inches='tight')

        
        
    def validate(self, Model):
    
        # Create an object to use utilities
        processor = Utilities()
        
        # Load the model
        Model.model = keras.models.load_model(Model.model_name+'/'+Model.model_id+'/'+Model.model_id+'.keras')

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in?)
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_name, Model.n_jets)
        
        # Load maxmean
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)

        # Pre-process the data
        totalX_jets, totalX_other, totalY, self.X_scaled_keys, self.Y_scaled_keys = processor.prepData(self.train_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)

        # Set how the data will be split (70 for training, 15 for validation, 15 for testing)
        split = 70/85   # Gave 85% to train file, now want 70% for the actual training
        _, valX_jets, _, valX_other, _, valY = train_test_split(totalX_jets, totalX_other, totalY, train_size=split)
        
        # Predictions and truth BEFORE they're back to the original scale
        print('Making predictions ...')
        preds_scaled = Model.model.predict([valX_jets, valX_other])
        true_scaled = valY 
        
        # Invert scaling
        scaler = normalize_new.Scaler()
        preds_origscale_dic = scaler.invscale_arrays(preds_scaled, self.Y_scaled_keys, Y_maxmean_dic)
        true_origscale_dic = scaler.invscale_arrays(true_scaled, self.Y_scaled_keys, Y_maxmean_dic)
        
        # Create basic plots of results and save them
        # Create directory for saving things in if it doesn't exist
        save_loc = Model.model_name+'/'+Model.model_id+'/val_plots/'
        if not os.path.exists(save_loc):
            os.mkdir(save_loc)
        
        # Scaled Plots
        if not os.path.exists(save_loc+'scaled/'):
            os.mkdir(save_loc+'scaled/')
        for i, key in enumerate(self.Y_scaled_keys):
            print(key)
            self.plot(Model.model_name, key, preds_scaled[:,i], true_scaled[:,i], save_loc+'scaled/')
            
        # Original Scale Plots
        if not os.path.exists(save_loc+'original/'):
            os.mkdir(save_loc+'original/')
        for key in preds_origscale_dic.keys():
            print(key)
            self.plot(Model.model_name, key, preds_origscale_dic[key], true_origscale_dic[key], save_loc+'original/')
            
        # Jet CM Plot (For JetPretrainer)
        if Model.model_name == 'JetPretrainer':
            self.jet_cm_plot(preds_scaled, true_scaled, save_loc+'scaled/')
            #self.jet_cm_plot(preds_origscale_dic, true_origscale_dic, save_loc+'original/')
            
            
            
# ------ MAIN CODE ------ #

if __name__ == "__main__":

    # Set up argument parser
    parser = ArgumentParser()

    # These arguments we want to be able to use for all of them
    parser = ArgumentParser(add_help=False)
    parser.add_argument('--model_name', help='Name of the model to be trained.', type=str, required=True, choices=['TRecNet','TRecNet+ttbar','TRecNet+ttbar+JetPretrain','JetPretrainer','TRecNet+ttbar+JetPretrainUnfrozen'])
    parser.add_argument('--model_id', help="Unique save name for the trained model.", type=str, required=True)
    parser.add_argument('--data', help="Path and file name for the training data to be used (e.g. '/mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_6j_train.h5').", type=str, required=True)
    parser.add_argument('--xmaxmean', help="Path and file name for the X_maxmean file to be used (e.g. '/home/jchishol/TRecNet/X_maxmean_variables_ttbar_ljets_6j_train.npy')", type=str, required=True)
    parser.add_argument('--ymaxmean', help="Path and file name for the Y_maxmean file to be used (e.g. '/home/jchishol/TRecNet/Y_maxmean_variables_ttbar_ljets_6j_train.npy')", type=str, required=True)
    
    # Parse arguments
    args = parser.parse_args()

    # Validate the model
    print('Beginning validation for '+args.model_name+'...')
    jn = args.model_id.split('jets')[0].split('_')[-1]
    Model = TRecNet_Model(args.model_name, model_id=args.model_id,n_jets=jn)
    Validator = Validating(args.data, args.xmaxmean, args.ymaxmean)
    Validator.validate(Model)


print('done :)')