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



class Testing:

    def __init__(self, test_file, xmm_file, ymm_file, data_type):
        self.test_file = test_file
        self.xmm_file = xmm_file
        self.ymm_file = ymm_file
        self.data_type = data_type
        self.X_keys = None
        self.Y_keys = None
        self.X_scaled_keys = None
        self.Y_scaled_keys = None

    def calculateExtraVariables(self, model_name, preds, eventNumbers):

        print('Calculating extra variables ...')
        
        # Create the 4-vectors for hadronic and leptonic tops
        th_vec = vector.arr({"pt": preds['th_pt'], "phi": preds['th_phi'], "eta": preds['th_eta'],"mass": preds['th_m']})
        tl_vec = vector.arr({"pt": preds['tl_pt'], "phi": preds['tl_phi'], "eta": preds['tl_eta'],"mass": preds['tl_m']}) 
        
        # Create 4-vector for ttbar (and get ttbar variables if necessary)
        if '+ttbar' in model_name:
            ttbar_vec = vector.arr({"pt": preds['ttbar_pt'], "phi": preds['ttbar_phi'], "eta": preds['ttbar_eta'],"mass": preds['ttbar_m']})
        else:
            ttbar_vec = th_vec + tl_vec
            preds['ttbar_pt'] = ttbar_vec.pt
            preds['ttbar_phi'] = ttbar_vec.phi
            preds['ttbar_eta'] = ttbar_vec.eta
            preds['ttbar_m'] = ttbar_vec.m
        
        # Get vector package to calculate the energies
        preds['th_E'] = th_vec.energy
        preds['tl_E'] = tl_vec.energy
        preds['ttbar_E'] = ttbar_vec.energy
        
        # Get vector package to calculate the rapidities
        preds['th_y'] = th_vec.rapidity
        preds['tl_y'] = tl_vec.rapidity
        preds['ttbar_y'] = ttbar_vec.rapidity
        
        # Calculate pout for th and tl
        ez = vector.arr({"px":np.array([0]*preds['th_eta'].shape[0]),"py":np.array([0]*preds['th_eta'].shape[0]),"pz":np.array([1]*preds['th_eta'].shape[0])})
        thP_vec = vector.arr({"px":th_vec.px,"py":th_vec.py,"pz":th_vec.pz})
        tlP_vec = vector.arr({"px":tl_vec.px,"py":tl_vec.py,"pz":tl_vec.pz})
        preds['th_pout'] = -thP_vec.dot(tlP_vec.cross(ez)/(tlP_vec.cross(ez).mag))   # For some reason this calculation is off by a negative compared to the truth ...
        preds['tl_pout'] = -tlP_vec.dot(thP_vec.cross(ez)/(thP_vec.cross(ez).mag))

        # Calculate some other ttbar variables
        preds['ttbar_dphi'] = abs(th_vec.deltaphi(tl_vec))
        preds['ttbar_Ht'] = preds['th_pt']+preds['tl_pt']
        preds['ttbar_yboost'] = 0.5*(preds['th_y']+preds['tl_y'])
        preds['ttbar_ystar'] = 0.5*(preds['th_y']-preds['tl_y'])
        preds['ttbar_chi'] = np.exp(2*np.abs(preds['ttbar_ystar']))
        
        preds['eventNumber'] = eventNumbers

        return preds
    


    def save_results(self, Model, preds_df, truths_df, save_loc):

        print('Saving results ...')

        save_name = save_loc+Model.model_id+'_'+self.data_type+'_Results.root'
        results_file = uproot.recreate(save_name)

        if self.data_type=='nominal': 
            results_file["reco"] = {key:preds_df[key] for key in list(preds_df.keys())}
            results_file["parton"] = {key:truths_df[key] for key in list(truths_df.keys())}
        else:
            results_file[self.data_type] = {key:preds_df[key] for key in list(preds_df.keys())}

        print('Results saved in %s.' % save_name)




    def test(self, Model, save_loc):

        # Create an object to use utilities
        processor = Utilities()
        
        # Load the things we'll need
        X_maxmean_dic, Y_maxmean_dic = processor.loadMaxMean(self.xmm_file, self.ymm_file)
        Model.model = keras.models.load_model(Model.model_name+'/'+Model.model_id+'/'+Model.model_id+'.keras')

        # These are the keys for what we're feeding into the pre-processing, and getting back in the end
        # X and Y variables to be used (NOTE: later have option to feed these in)
        self.X_keys, self.Y_keys = processor.getInputKeys(Model.model_name,Model.n_jets)

        # Pre-process the data
        testX_jets, testX_other, testY, self.X_scaled_keys, self.Y_scaled_keys = processor.prepData(self.test_file, X_maxmean_dic, Y_maxmean_dic, self.X_keys, self.Y_keys, Model.n_jets, Model.mask_value)

        # Predictions and truth BEFORE they're back to the original scale
        preds_scaled = Model.model.predict([testX_jets, testX_other])
        true_scaled = testY 

        # Invert scaling
        preds_origscale_dic, true_origscale_dic = processor.unscale(preds_scaled, true_scaled, self.Y_scaled_keys, Y_maxmean_dic)


        # Get the true values
        with h5py.File(self.test_file,'r') as dataset:
            eventNumbers = np.array(dataset.get('eventNumber'))
            if self.data_type=='nominal': 
                truth_keys = ['eventNumber','jet_n']+list(filter(lambda a: 'th_' in a or 'tl_' in a or 'ttbar_' in a or 'wh_' in a or 'wl_' in a, dataset.keys()))  
                truths_df = pd.DataFrame({key:dataset.get(key) for key in truth_keys})
            else:
                truths_df = None  # If working with systematics, we have no truth
                
        # Calculate all the same variables we had for truth, using the predicted values
        preds_df = pd.DataFrame(preds_origscale_dic)
        preds_df = self.calculateExtraVariables(Model.model_name, preds_df, eventNumbers)

        # Save results
        self.save_results(Model, preds_df, truths_df)
        




# ------ MAIN CODE ------ #

# Set up argument parser
parser = ArgumentParser()
parser.add_argument('--model_name', help='Name of the model to be trained.', type=str, required=True, choices=['TRecNet','TRecNet+ttbar','TRecNet+ttbar+JetPretrain','JetPretrainer','TRecNet+ttbar+JetPretrainUnfrozen'])
parser.add_argument('--data', help="Path and file name for the training data to be used (e.g. '/mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_6j_train.h5').", type=str, required=True)
parser.add_argument('--xmaxmean', help="Path and file name for the X_maxmean file to be used (e.g. '/home/jchishol/TRecNet/X_maxmean_variables_ttbar_ljets_6j_train.npy')", type=str, required=True)
parser.add_argument('--ymaxmean', help="Path and file name for the Y_maxmean file to be used (e.g. '/home/jchishol/TRecNet/Y_maxmean_variables_ttbar_ljets_6j_train.npy')", type=str, required=True)
parser.add_argument('--model_id', help="Unique save name for the trained model.", type=str, required=True)
parser.add_argument('--data_type', choices=['nominal','sysUP','sysDOWN'],help='Type of data.', required=True)
parser.add_argument('--save_loc', help="Directory in which to save the results (e.g. '/mnt/xrootdg/jchishol/mntuples_08_01_22/Results/').", type=str, required=True)

# Parse arguments
args = parser.parse_args()

# Test the model
print('Beginning testing for '+args.model_name+'...')
jn = args.model_name.split('jets')[0].split('_')[-1]
Model = TRecNet_Model(args.model_name, model_id=args.model_id,n_jets=jn)
Tester = Testing(args.data, args.xmaxmean, args.ymaxmean, args.data_type)
Tester.test(Model, args.save_loc)


print('done :)')