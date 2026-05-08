import os, sys
sys.path.append("/home/jchishol/TRecNet")
sys.path.append("home/jchishol/")
os.environ["CUDA_VISIBLE_DEVICES"]="1"    # These are the GPUs visible for training

import numpy as np
import itertools
import matplotlib
matplotlib.use('Agg')
import h5py

#from clr_callback import * 

import Scaler
import ShapeTimesteps


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

    def getInputKeys(self, model_v, n_jets, add_ttbar, b_mode):
        """
        Gets lists of the (original scale) X and Y variable keys.

            Parameters:
                model_v (str): Version of the model (e.g. 'TRecNet_ttbb_v1').
                n_jets (int): Number of jets the model is or will be trained on.
                add_ttbar (bool): Whether or not to include ttbar variables.
                b_mode (str): How b's from ttbar decay are defined ('bbbar' or 'b1b2').
                                
            Returns:
                X_keys (list of str): Keys for the (original scale) X variables.
                Y_keys (list of str): Keys for the (original scale) Y variables.
        """

        # X keys (always the same)
        X_keys = ['j'+str(i+1)+'_'+v for i, v in itertools.product(range(n_jets),['pt','eta','phi','m','isbtag'])] + ['lep_pt', 'lep_eta', 'lep_phi', 'met_met', 'met_phi']
            
        # Y keys
        if 'JetPretrainer' in model_v: 
            Y_keys = ['j'+str(i+1)+'_isFromttbar' for i in range(n_jets)]
            
        elif 'bbPretrainer' in model_v:
            Y_keys = ['j'+str(i+1)+'_isExtraB' for i in range(n_jets)]
            
        else:
            # Start with hadronic keys
            Y_keys = ['th_pt', 'th_eta','th_phi','th_m', 'wh_pt', 'wh_eta', 'wh_phi', 'wh_m'] 
            
            # Next is ttbar keys (if included), since these are output with the hadronic keys
            if add_ttbar:
                Y_keys.extend(['ttbar_pt','ttbar_eta','ttbar_phi','ttbar_m'])
                
            # Leptonic keys AFTER hadronic + ttbar keys
            Y_keys.extend(['tl_pt', 'tl_eta', 'tl_phi', 'tl_m', 'wl_pt', 'wl_eta', 'wl_phi', 'wl_m']) # lep keys AFTER hadronic + ttbar keys

            # bbbar or b1b2 keys at the end
            if b_mode == 'bbbar':
                Y_keys.extend(['b_t_pt','b_t_m','b_t_eta','b_t_phi','bbar_tbar_pt','bbar_tbar_m','bbar_tbar_eta','bbar_tbar_phi'])
            elif b_mode == 'b1b2':
                Y_keys.extend(['b1_pt','b1_m','b1_eta','b1_phi','b2_pt','b2_m','b2_eta','b2_phi'])

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

        scaler = Scaler.Scaler()
        X_df = scaler.scale_arrays(dataset, X_keys, X_maxmean_dic)
        scaled_X_keys = X_df.keys()
        
        if onlyX:
            Y_df = None
            scaled_Y_keys = scaler.get_scaled_Ykeys(Y_keys)
        else:
            Y_df = scaler.scale_arrays(dataset, Y_keys, Y_maxmean_dic)
            scaled_Y_keys = Y_df.keys()

        return X_df, Y_df, scaled_X_keys, scaled_Y_keys



    def scale_and_shape(self, datafile, X_maxmean_dic, Y_maxmean_dic, X_keys, Y_keys, jn, mask_value, onlyX=False):
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
        
        with h5py.File(datafile,'r') as dataset:   # Only want the dataset open as long as we need it
            
            # Create the timestep builder while we still have the dataset open
            timestep_builder = ShapeTimesteps.ShapeTimesteps(dataset, jn, mask_value)
            
            # Scales data set to be between -1 and 1, with a mean of 0, and encodes phi in other variables (e.g. px, py)
            X_df, Y_df, scaled_X_keys, scaled_Y_keys = self.scale(dataset, X_keys, Y_keys, X_maxmean_dic, Y_maxmean_dic, onlyX)

        # Split up jets and other for X, and Y just all stays together
        totalX_jets, totalX_other = timestep_builder.reshape_X(X_df)
        Y_total = np.array(Y_df)
        return totalX_jets, totalX_other, Y_total, scaled_X_keys, scaled_Y_keys
