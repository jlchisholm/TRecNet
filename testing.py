import numpy as np
import vector
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import h5py
import os, sys, argparse
from argparse import ArgumentParser
from clr_callback import *
from tensorflow.keras.optimizers import * 
import glob
import re
import importlib 
import uproot
os.environ["CUDA_VISIBLE_DEVICES"]="1"


class testing:
    def getTestResults(model_name,data_type):


        print('Loading model and data...')

        # Numpy array of [max,mean] for each variable
        X_maxmean_dic = np.load('X_maxmean_variables_ttbar_ljets_jetMatch04_6jets_train.npy',allow_pickle=True).item()  # should always use this same file I think?
        Y_maxmean_dic = np.load('Y_maxmean_variables_ttbar_ljets_jetMatch04_6jets_train.npy',allow_pickle=True).item()

        # Dataset
        if data_type=='full':
            dataset = h5py.File('/mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_jetMatch04_6jets_test.h5','r')
        elif data_type=='sysUP' or 'sysDOWN':
            dataset = h5py.File('/mnt/xrootdg/jchishol/mntuples_08_01_22/variables_ttbar_ljets_'+data_type+'_test.h5','r')

        # Model
        model = keras.models.load_model(model_name+'/'+model_name+'_full.keras')


        # Number of events in dataset
        size = np.array(dataset.get('met_met')).size
        crop0 = size


        # X and Y variables to be used
        X_keys = ['j1_pt', 'j1_eta', 'j1_phi', 'j1_m', 'j1_isbtag', 'j2_pt', 'j2_eta', 'j2_phi', 'j2_m', 'j2_isbtag', 'j3_pt', 'j3_eta', 'j3_phi', 'j3_m', 'j3_isbtag', 'j4_pt', 'j4_eta', 'j4_phi', 'j4_m', 'j4_isbtag', 'j5_pt', 'j5_eta', 'j5_phi', 'j5_m', 'j5_isbtag', 'j6_pt', 'j6_eta', 'j6_phi', 'j6_m', 'j6_isbtag', 'lep_pt', 'lep_eta', 'lep_phi', 'met_met', 'met_phi']
        Y_keys = ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m', 'tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']
        if '+ttbar' in model_name: 
            Y_keys = Y_keys + ['ttbar_pt','ttbar_eta','ttbar_phi','ttbar_m']
        if 'JetPretrain' in model_name: 
            Y_keys = ['j1_isTruth','j2_isTruth','j3_isTruth','j4_isTruth','j5_isTruth','j6_isTruth'] + Y_keys

        # Variable keys, needed for helper codes
        phi_keys = list(filter(lambda a: 'phi' in a, dataset.keys()))
        eta_keys = list(filter(lambda a: 'eta' in a, dataset.keys()))
        pt_keys =  list(filter(lambda a: 'pt' in a, dataset.keys()))
        m_keys = list(filter(lambda a: 'm' in a, dataset.keys()))
        isbtag_keys = list(filter(lambda a: 'isbtag' in a, dataset.keys()))

        # Get the maxmean values we actually need for this particular model
        xmm_keys = list(X_maxmean_dic.keys())
        if model_name == 'Model_Custom':
            ymm_keys = [key for key in Y_maxmean_dic.keys() if 'isTruth' not in key and 'ttbar' not in key]
        elif model_name == 'Model_Custom+ttbar':
            ymm_keys = [key for key in Y_maxmean_dic.keys() if 'isTruth' not in key]
        else:
            ymm_keys = list(Y_maxmean_dic.keys())
        X_maxmean = np.array([X_maxmean_dic[key] for key in xmm_keys])
        Y_maxmean = np.array([Y_maxmean_dic[key] for key in ymm_keys])



        # Need to reload these since we've imported the dataset
        import normalize
        import shape_timesteps
        importlib.reload(normalize)
        importlib.reload(shape_timesteps)



        print('Scaling inputs ...')

        # Scales data set to be between -1 and 1, with a mean of 0
        Scaler = normalize.Scale_variables(phi_keys,dataset,crop0)   # Need to reorganize the helper codes this is so stupid
        X_total, X_names = Scaler.scale_arrays(dataset,X_keys, X_maxmean,crop0)
        if data_type=='full': Y_total, Y_names = Scaler.scale_arrays(dataset,Y_keys, Y_maxmean,crop0)  # Is this necessary actually?



        # Kind of annoying that we're using ynames from the helper code, seems unnecessary
        if data_type=='full':
            if model_name=='Model_JetPretrain+ttbar':
                # And don't want jet pretraining variables later
                Y_total = Y_total[:,6:]
                Y_names = Y_names[6:]
                Y_maxmean = Y_maxmean[6:]
        else:
            if model_name=='Model_JetPretrain+ttbar':
                # And don't want jet pretraining variables later
                Y_names = ymm_keys[6:]
                Y_maxmean = Y_maxmean[6:]
            else:
                Y_names = ymm_keys




        if data_type=='full': testY = Y_total   # Need?

        # Split up jets and other for X
        timestep_builder = shape_timesteps.Shape_timesteps()
        testX_jets, testX_other = timestep_builder.reshape_X(dataset,X_total, X_names, phi_keys, crop0, False,True)




        # Define bins (I guess we need for helper codes? again, something we should see if we can cut)
        other_bins = np.linspace(-1, 1, 40)
        phi_bins = np.linspace(-1, 1, 40)
        pt_bins = np.linspace(-1, 1, 40)
        Y_bins = [phi_bins if 'phi' in name else pt_bins if 'pt' in name else other_bins for name in Y_names]

        print('Making predictions ...')

        # Predictions and truth BEFORE they're back to the original scale
        predictions_unscaled = model.predict([testX_jets, testX_other])
        if data_type=='full': true_unscaled = testY  # Need?

        predictions_origscale = Scaler.invscale_arrays(predictions_unscaled, Y_names, Y_maxmean)
        if data_type=='full': true_origscale = Scaler.invscale_arrays(true_unscaled, Y_names, Y_maxmean) # Need?







        

        if model_name=='Model_JetPretrain+ttbar':
            pred_keys = Y_keys[6:]
        else:
            pred_keys = Y_keys

        # We're going to turn this into dataframes for easier reading
        if data_type=='full':
            truth_keys = list(dataset.keys())[43:-8]
            truth_keys.extend(('eventNumber','jet_n'))
            truths = pd.DataFrame({key:dataset.get(key) for key in truth_keys})  # or something like this
        preds = pd.DataFrame(predictions_origscale,columns=pred_keys)


        # Gotta calculate some stuff!
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
        
        preds['eventNumber'] = dataset.get('eventNumber')
        preds['jet_n'] = dataset.get('jet_n')
        

        print('Saving results ...')

        # Save as a tree root file
        results_file = uproot.recreate('/mnt/xrootdg/jchishol/'+model_name+'_'+data_type+'_Results.root')
        results_file["reco"] = {key:preds[key] for key in list(preds.keys())}
        if data_type=='full': results_file["parton"] = {key:truths[key] for key in list(truths.keys())}





# ---------- GET ARGUMENTS FROM COMMAND LINE ---------- #

# Create the main parser and subparsers
parser = ArgumentParser()
parser.add_argument('--model',choices=['Model_Custom', 'Model_Custom+ttbar', 'Model_JetPretrain+ttbar'],help='TRecNet version you would like to use.', required=True)
parser.add_argument('--type',choices=['full','sysUP','sysDOWN'],help='Type of data.', required=True)

# Parse the arguments and proceed with stuff
args = parser.parse_args()
print('made it here at least')
testing.getTestResults(args.model,args.type)

print('done :)')