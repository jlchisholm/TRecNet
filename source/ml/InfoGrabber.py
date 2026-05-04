#########################################################################
#                                                                       #
#  InfoGrabber.py.                                                      #
#  Author: Jenna Chisholm                                               #
#  Updated: Sept.8/25                                                   #
#                                                                       #
#  Defines classes and functions to help get info from trained models.  # 
#                                                                       #
#  Thoughts for improvements: Have X and Y keys as input variables?     #
#
#
# Updated Sep.25/25
#
#                                                                       #
#########################################################################

import sys, os
import glob
from paths import resolve_model_dir

class InfoGrabber:
    """
    A class for using pre-existing neural networks.
    """  
    
    def __init__(self):
        """
        Initializes an InfoGrabber object. 
        """

        
    def check_model_trained(self, model_id):
        """
        Checks that the input model id actually corresponds to a trained model.
        
            Parameters:
                model_id (str): ID of the trained model.
        """
        # now uses the new path.py utilities (will sys.exit if not found)
        _ = self.resolve_model_dir(model_id)
        
    def resolve_model_dir(self, model_id: str) -> str:
        '''New wrapper around the shared utility'''
        # Uses path.py utility to resolve model directory 
        return resolve_model_dir(model_id)


    def get_train_data_file(self, model_id):
        """
        Extracts the training data h5 file name from the training info.

            Parameters:
                model_id (str): ID of the trained model.
                
            Returns:
                train_data_file (str): Name (including path) of the h5 train data file.
        """
        # get model directory with path.py utility
        model_dir = self.resolve_model_dir(model_id)

        # Now searches for info/run_Info.txt and falls back to {model_id}_Info.txt
        new_info = os.path.join(model_dir, 'info', 'run_Info.txt')
        old_info = os.path.join(model_dir, f'{model_id}_Info.txt')  # backward-compat fallback
        info_file = new_info if os.path.isfile(new_info) else old_info

        train_data_file = None
        # find data file
        with open(info_file) as file:
            for line in file:
                if 'Training Data File: ' in line:
                    # success
                    train_data_file = line.split('Training Data File: ')[1].strip()
                    break
        if train_data_file is None:
            # failure, exit
            print('Failed to find train data. Program exiting.')
            sys.exit()
        return train_data_file
    
    
    def get_maxmean_files(self, model_id):
        """
        Extracts the maxmean file names from the training info.

            Parameters:
                model_id (str): ID of the trained model.
                
            Returns:
                xmm_file (str): Name (including path) of the xmaxmean file. 
                ymm_file (str): Name (including path) of the ymaxmean file.           
        """
        model_dir = self.resolve_model_dir(model_id)
        # prefers a scaling/ subdirectory if present
        scaling_dir = os.path.join(model_dir, "scaling")
        if os.path.isdir(scaling_dir):
            files = os.listdir(scaling_dir)
            xmm = next((f for f in files if 'X_maxmean' in f), None)
            ymm = next((f for f in files if 'Y_maxmean' in f), None)
            if xmm and ymm:
                return os.path.join(scaling_dir, xmm), os.path.join(scaling_dir, ymm)
        # if legacy uses model_dir directly
        files = os.listdir(model_dir)
        xmm = next((f for f in files if 'X_maxmean' in f), None)
        ymm = next((f for f in files if 'Y_maxmean' in f), None)
        if xmm is None or ymm is None:
            # failure, exit
            print('Failed to find maxmean files. Something must be terribly wrong. Exiting program.')
            sys.exit()
        return os.path.join(model_dir, xmm), os.path.join(model_dir, ymm)
        
    def get_train_val_split(self, model_id):
        """
        Extracts the training data h5 file name from the training info.

            Parameters:
                model_id (str): ID of the trained model.
                
            Returns:
                split (float): Percentage of data from training file that will be given to training, while the remainder is used for validation (taken from config file).
        """

        model_dir = self.resolve_model_dir(model_id)
        # same dual-path detection of info/run_Info.txt
        new_info = os.path.join(model_dir, 'info', 'run_Info.txt')
        old_info = os.path.join(model_dir, f'{model_id}_Info.txt')
        info_file = new_info if os.path.isfile(new_info) else old_info

        value = None
        with open(info_file) as f:
            for line in f:
                if 'Percentage of Train Data Used for Training:' in line:
                    value = line.split('Percentage of Train Data Used for Training:')[1].strip()
                    break
        if value is None:
            print('Failed to find train/val split. Program exiting.')
            sys.exit()
        # added try/except to make function defensive
        # added cases for detrmining split:
        # two cases:
        #  a) "0.7" or "0.70" -> float
        #  b) "[70,15,15]" -> use 70/sum -> 0.7
        try:
            return float(value)
        # try to parse as list
        except ValueError:
            txt = value.strip()
            if txt.startswith('[') and txt.endswith(']'):
                parts = [float(x) for x in txt.strip('[]').replace(',', ' ').split()]
                if len(parts) >= 1 and sum(parts) > 0:
                    return parts[0] / sum(parts)
            print('Could not parse training split. Exiting program.')
            sys.exit()
    
    
    def get_data_type(self, data_file):
        """
        Extracts the data type ('nom', 'sysUP', or 'sysDOWN' from the data file name).
        
            Parameters:
                data_file (str): Name (including path) of the data file.
            Returns:
                data_type (str): Type of data for the data file ('nom', 'sysUP', or 'sysDOWN').
        """
            
        if 'nom' in data_file:
            data_type = 'nom' 
        elif 'sysUP' in data_file:
            data_type = 'sysUP'
        elif 'sysDOWN' in data_file:
            data_type = 'sysDOWN'
        else:
            print('Failed to figure out data type from file name. Exiting program.')
            sys.exit()
            
        return data_type
    
    def get_model_v(self, model_id):
        
        model_v = model_id.split('v')[0] + 'v' + model_id.split('v')[1].split('_')[0]
        return model_v
    
    def get_njets(self, model_id):
        
        n_jets = model_id.split('jets')[0].split('_')[-1]
        return int(n_jets)
    
    def get_ttbar_status(self, model_id):
        
        add_ttbar = True if '+ttbar' in model_id else False
        return add_ttbar
    
    def get_b_mode(self, model_id):
        
        b_mode = 'bbbar' if 'bbbar' in model_id else 'b1b2' if 'b1b2' in model_id else None
        return b_mode